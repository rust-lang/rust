//@ run-fail

//! This test ensures that `#[track_caller]` can be applied directly to diverging functions, as
//! the tracking issue says: https://github.com/rust-lang/rust/issues/47809#issue-292138490.
//! Because the annotated function must diverge and a panic keeps that faster than an infinite loop,
//! we don't inspect the location returned -- it would be difficult to distinguish between the
//! explicit panic and a failed assertion. That it compiles and runs is enough for this one.

#[track_caller]
fn doesnt_return() -> ! {
    let _location = core::panic::Location::caller();
    panic!("huzzah");
}

fn main() {
    doesnt_return();
}
