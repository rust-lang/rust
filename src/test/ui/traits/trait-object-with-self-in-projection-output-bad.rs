// Regression test for #56288. Checks that if a supertrait defines an associated type
// projection that references `Self`, then that associated type must still be explicitly
// specified in the `dyn Trait` variant, since we don't know what `Self` is anymore.

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

trait ConstI32 {
    type Out;
}

impl<T: ?Sized> ConstI32 for T {
    type Out = i32;
}

// Test that you still need to manually give a projection type if the Output type
// is normalizable.
trait NormalizableHelper:
    Base<Output=<Self as ConstI32>::Out>
{
    type Target;
}

impl NormalizableHelper for u32
{
    type Target = i32;
}

fn main() {
    let _x: Box<dyn Helper<Target=i32>> = Box::new(2u32);
    //~^ ERROR the value of the associated type `Output` (from the trait `Base`) must be specified

    let _y: Box<dyn NormalizableHelper<Target=i32>> = Box::new(2u32);
    //~^ ERROR the value of the associated type `Output` (from the trait `Base`) must be specified
}
