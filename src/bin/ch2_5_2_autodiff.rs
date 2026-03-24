use tch::Tensor;

fn main() {
    let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0]).set_requires_grad(true);
    let y = &x * &x;
    y.sum(tch::Kind::Float).backward();
    println!("梯度结果：{:?}", x.grad());

    let _ = x.grad().zero_();
    println!("清零后的梯度：{:?}", x.grad());
}
