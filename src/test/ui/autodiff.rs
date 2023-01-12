// Check autodiff attribute
// edition:2018

extern "C" fn rosenbrock(a: f32, b: f32, x: f32, y: f32) -> f32 {
    let (z, w) = (a-x, y-x*x);

    z*z + b*w*w
}

#[autodiff(rosenbrock, mode = "forward")]
extern "C" {
    fn dx_rosenbrock(a: f32, b: f32, x: f32, y: f32, d_x: &mut f32);
    fn dy_rosenbrock(a: f32, b: f32, x: f32, y: f32, d_y: &mut f32);
}

fn main() {}
