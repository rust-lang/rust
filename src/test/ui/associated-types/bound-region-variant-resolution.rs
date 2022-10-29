// This is a bug!
// We should emit the same error message regardless of
// whether the projection type contains bound regions.

// check-fail

fn test<'x>() {
    use std::ops::Deref;
    None::<for<'y> fn(<&'x Option<()> as Deref>::Target::Some)>;
    //~^ ERROR expected type, found variant
    None::<for<'y> fn(<&'y Option<()> as Deref>::Target::Some)>;
    //~^ ERROR ambiguous associated type
}

fn main() {}
