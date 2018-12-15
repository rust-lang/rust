// compile-pass

trait Base {
    type Output;
}

trait Helper: Base<Output=<Self as Helper>::Target> {
    type Target;
}

impl Base for u32
{
    type Output = i32;
}

impl Helper for u32
{
    type Target = i32;
}

fn main() {
    let _x: Box<dyn Helper<Target=i32, Output=i32>> = Box::new(2u32);
}
