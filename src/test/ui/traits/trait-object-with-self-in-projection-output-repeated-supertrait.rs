// build-pass (FIXME(62277): could be check-pass?)

// FIXME(eddyb) shorten the name so windows doesn't choke on it.
#![crate_name = "trait_test"]

// Regression test related to #56288. Check that a supertrait projection (of
// `Output`) that references `Self` is ok if there is another occurence of
// the same supertrait that specifies the projection explicitly, even if
// the projection's associated type is not explicitly specified in the object type.
//
// Note that in order for this to compile, we need the `Self`-referencing projection
// to normalize fairly directly to a concrete type, otherwise the trait resolver
// will hate us.
//
// There is a test in `trait-object-with-self-in-projection-output-bad.rs` that
// having a normalizing, but `Self`-containing projection does not *by itself*
// allow you to avoid writing the projected type (`Output`, in this example)
// explicitly.

trait ConstI32 {
    type Out;
}

impl<T: ?Sized> ConstI32 for T {
    type Out = i32;
}

trait Base {
    type Output;
}

trait NormalizingHelper: Base<Output=<Self as ConstI32>::Out> + Base<Output=i32> {
    type Target;
}

impl Base for u32
{
    type Output = i32;
}

impl NormalizingHelper for u32
{
    type Target = i32;
}

fn main() {
    // Make sure this works both with and without the associated type
    // being specified.
    let _x: Box<dyn NormalizingHelper<Target=i32>> = Box::new(2u32);
    let _y: Box<dyn NormalizingHelper<Target=i32, Output=i32>> = Box::new(2u32);
}
