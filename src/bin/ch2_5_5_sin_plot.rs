use plotters::prelude::*;
use tch::Tensor;

fn autodiff_deriv(x: f64) -> f64 {
    let t = Tensor::from_slice(&[x]).set_requires_grad(true);
    t.sin().backward();
    t.grad().double_value(&[])
}

fn numerical_deriv(x: f64, h: f64) -> f64 {
    let y1 = Tensor::from_slice(&[x + h]).sin().double_value(&[]);
    let y2 = Tensor::from_slice(&[x - h]).sin().double_value(&[]);
    (y1 - y2) / (2.0 * h)
}

fn main() {
    let pi = std::f64::consts::PI;
    let x_range: Vec<f64> = (-200..=200).map(|i| i as f64 * pi / 100.0).collect();

    let sin_vals: Vec<f64> = x_range.iter().map(|&x| x.sin()).collect();
    let auto_vals: Vec<f64> = x_range.iter().map(|&x| autodiff_deriv(x)).collect();
    let num_vals: Vec<f64> = x_range.iter().map(|&x| numerical_deriv(x, 0.001)).collect();

    let root = BitMapBackend::new("sin_and_derivative.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let subplots = root.split_evenly((2, 1));

    {
        let mut chart = ChartBuilder::on(&subplots[0])
            .caption("f(x) = sin(x)", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-2.0 * pi..2.0 * pi, -1.2..1.2)
            .unwrap();
        chart.configure_mesh().draw().unwrap();
        chart
            .draw_series(LineSeries::new(x_range.iter().zip(sin_vals.iter()).map(|(&x, &y)| (x, y)), &BLUE))
            .unwrap()
            .label("sin(x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
        chart.configure_series_labels().draw().unwrap();
    }

    {
        let mut chart = ChartBuilder::on(&subplots[1])
            .caption("Derivative of sin(x)", ("sans-serif", 25).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-2.0 * pi..2.0 * pi, -1.2..1.2)
            .unwrap();
        chart.configure_mesh().draw().unwrap();
        chart
            .draw_series(LineSeries::new(x_range.iter().zip(auto_vals.iter()).map(|(&x, &y)| (x, y)), &RED))
            .unwrap()
            .label("AutoDiff")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        chart
            .draw_series(LineSeries::new(x_range.iter().zip(num_vals.iter()).map(|(&x, &y)| (x, y)), &GREEN))
            .unwrap()
            .label("Numerical (h=0.001)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        chart.configure_series_labels().draw().unwrap();
    }

    println!("Plot saved to sin_and_derivative.png");
}
