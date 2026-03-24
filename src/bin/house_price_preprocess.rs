use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use std::collections::{HashMap, HashSet};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = "房价预测/train.csv";
    let out_path = "房价预测/train_processed.csv";

    let mut reader = ReaderBuilder::new().from_path(path)?;
    let headers = reader.headers()?.clone();
    let records: Vec<StringRecord> = reader.records().filter_map(|r| r.ok()).collect();

    println!("样本数：{}, 特征数：{}\n", records.len(), headers.len());

    let missing = count_missing(&headers, &records);
    let with_missing: Vec<_> = missing.iter().filter(|(_, &c)| c > 0).collect();
    println!("有缺失的特征 ({} 个):", with_missing.len());
    for (f, c) in &with_missing {
        println!("  - {}: {} ({:.1}%)", f, c, (**c as f64 / records.len() as f64) * 100.0);
    }

    let categorical = find_categorical(&headers, &records);
    println!("\n离散特征：{} 个", categorical.len());

    let mut data: Vec<Vec<String>> = records.iter().map(|r| r.iter().map(String::from).collect()).collect();

    for (feat, _) in &with_missing {
        let idx = headers.iter().position(|h| h == *feat).unwrap();
        if categorical.contains(*feat) {
            fill_mode(&mut data, idx);
        } else {
            fill_median(&mut data, idx);
        }
    }

    let new_headers: Vec<&str> = headers.iter().filter(|h| *h != "Id").collect();
    let id_idx = headers.iter().position(|h| h == "Id").unwrap();

    let mut writer = WriterBuilder::new().from_path(out_path)?;
    writer.write_record(&new_headers)?;
    for rec in &data {
        let row: Vec<&String> = rec.iter().enumerate().filter(|(i, _)| *i != id_idx).map(|(_, v)| v).collect();
        writer.write_record(&row)?;
    }

    println!("\n已保存到：{}", out_path);
    println!("处理后：{} 特征，{} 样本", new_headers.len(), data.len());
    Ok(())
}

fn count_missing(headers: &StringRecord, records: &[StringRecord]) -> HashMap<String, usize> {
    let mut cnt = HashMap::new();
    for h in headers.iter() { cnt.insert(h.to_string(), 0); }
    for rec in records {
        for (i, v) in rec.iter().enumerate() {
            if v.is_empty() || v == "NA" || v == "na" {
                *cnt.get_mut(headers.get(i).unwrap()).unwrap() += 1;
            }
        }
    }
    cnt
}

fn find_categorical(headers: &StringRecord, records: &[StringRecord]) -> HashSet<String> {
    let num_kw = ["SF", "Area", "Feet", "Lot", "Year", "Price", "Cost"];
    let mut cat = HashSet::new();

    for h in headers.iter() {
        if h == "Id" || h == "SalePrice" { continue; }
        if num_kw.iter().any(|k| h.contains(k)) { continue; }

        let idx = headers.iter().position(|x| x == h).unwrap();
        let uniq: HashSet<&str> = records.iter()
            .take(100)
            .filter_map(|r| r.get(idx))
            .filter(|v| !v.is_empty() && *v != "NA")
            .collect();

        if !uniq.iter().all(|v| v.parse::<f64>().is_ok()) || uniq.len() <= 10 {
            cat.insert(h.to_string());
        }
    }
    cat
}

fn fill_mode(data: &mut Vec<Vec<String>>, col: usize) {
    let mut cnt: HashMap<String, usize> = HashMap::new();
    for rec in data.iter() {
        let v = &rec[col];
        if !v.is_empty() && v != "NA" {
            *cnt.entry(v.clone()).or_insert(0) += 1;
        }
    }
    let mode = cnt.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or_else(|| "None".into());
    for rec in data.iter_mut() {
        let v = &rec[col];
        if v.is_empty() || *v == "NA" { rec[col] = mode.clone(); }
    }
}

fn fill_median(data: &mut Vec<Vec<String>>, col: usize) {
    let mut vals: Vec<f64> = data.iter()
        .filter_map(|r| r[col].parse::<f64>().ok())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if vals.is_empty() { 0.0 } else {
        let m = vals.len() / 2;
        if vals.len() % 2 == 0 { (vals[m-1] + vals[m]) / 2.0 } else { vals[m] }
    };
    for rec in data.iter_mut() {
        let v = &rec[col];
        if v.is_empty() || *v == "NA" { rec[col] = med.to_string(); }
    }
}
