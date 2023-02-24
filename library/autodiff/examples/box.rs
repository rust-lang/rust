#![feature(bench_black_box)]
use autodiff::autodiff;

#[autodiff_into]
fn sin(x: &Box<f32>, y: &Box<f32>) -> f32 {
    f32::sin(**x) + f32::tanh(**y)
}
#[autodiff_into(Reverse, Active, Duplicated, Duplicated)]
fn cos_box(
    x: &Box<f32>, d_x: &mut Box<f32>, 
    y: &Box<f32>, d_y: &mut Box<f32>, 
    fac: f32) {
    std::hint::black_box((sin(x, y), d_x, d_y, x, y, fac));
}

//#[autodiff(cos_box, Reverse, Active, Duplicated)]
//fn sin(x: &Box<f32>) -> f32 {
//    f32::sin(**x)
//}

fn main() {
    let x = Box::<f32>::new(3.14);
    let mut df_dx = Box::<f32>::new(0.0);
    let y = Box::<f32>::new(3.14);
    let mut df_dy = Box::<f32>::new(0.0);
    cos_box(&x, &mut df_dx, &y, &mut df_dy, 1.0);

    dbg!(&df_dx, &df_dy);

    assert!(*df_dx == f32::cos(*x));
}
