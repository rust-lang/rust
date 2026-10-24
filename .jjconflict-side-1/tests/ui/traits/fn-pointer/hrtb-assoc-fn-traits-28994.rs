//@ check-pass
//! Tests that a HRTB + FnOnce bound involving an associated type don't prevent
//! a function pointer from implementing `Fn` traits.
//! Test for <https://github.com/rust-lang/rust/issues/28994>

trait LifetimeToType<'a> {
    type Out;
}

impl<'a> LifetimeToType<'a> for () {
    type Out = &'a ();
}

fn id<'a>(val: &'a ()) -> <() as LifetimeToType<'a>>::Out {
    val
}

fn assert_fn<F: for<'a> FnOnce(&'a ()) -> <() as LifetimeToType<'a>>::Out>(_func: F) { }

fn main() {
    assert_fn(id);
}
