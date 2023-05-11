// Test that we are special casing "outlives" for opaque types.
//
// The return type of a closure is not required to outlive the closure. As such
// the following code would not compile if we used a standard outlives check
// when checking the return type, because the return type of the closure would
// be `&ReEmpty i32`, and we don't allow `ReEmpty` to occur in the concrete
// type used for an opaque type.
//
// However, opaque types are special cased to include check all regions in the
// concrete type against the bound, which forces the return type to be
// `&'static i32` here.

// build-pass (FIXME(62277): could be check-pass?)

fn make_identity() -> impl Sized {
    |x: &'static i32| x
}

fn make_identity_static() -> impl Sized + 'static {
    |x: &'static i32| x
}

fn main() {}
