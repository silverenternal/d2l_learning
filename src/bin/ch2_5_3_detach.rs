use tch::Tensor;

fn main() {
    let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0]).set_requires_grad(true);
    let y = &x * &x;
    let u = y.detach();
    let z = &u * &x;

    z.sum(tch::Kind::Float).backward();
    let mut grad_z_x = x.grad();
    println!("z 关于 x 的梯度 (u 视为常数): {:?}", grad_z_x);
    println!("验证是否等于 u: {:?}", grad_z_x == u);

    if grad_z_x.defined() {
        let _ = grad_z_x.zero_();
    }

    y.sum(tch::Kind::Float).backward();
    let grad_y_x = x.grad();
    let two_x = Tensor::from_slice(&[0.0f32, 2.0, 4.0, 6.0]);
    println!("\ny 关于 x 的梯度 (y = x*x): {:?}", grad_y_x);
    println!("验证是否等于 2*x: {:?}", grad_y_x == two_x);
}
