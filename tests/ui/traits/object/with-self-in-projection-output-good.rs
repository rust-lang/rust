//@ build-pass (FIXME(62277): could be check-pass?)

// Regression test related to #56288. Check that a supertrait projection (of
// `Output`) that references `Self` can be ok if it is referencing a projection (of
// `Self::Target`, in this case). Note that we still require the user to manually
// specify both `Target` and `Output` for now.

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
