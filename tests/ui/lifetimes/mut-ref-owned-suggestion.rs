//! Regression test for <https://github.com/rust-lang/rust/issues/150077>
//! Tests that `&mut T` suggests `T`, not `mut T`, when recommending an owned value.
fn with_fn(_f: impl Fn() -> &mut ()) {}
//~^ ERROR: missing lifetime specifier

fn with_fn_has_return(_f: impl Fn() -> &mut ()) -> i32 {
    //~^ ERROR: missing lifetime specifier
    2
}

fn with_dyn(_f: Box<dyn Fn() -> &mut i32>) {}
//~^ ERROR: missing lifetime specifier

fn trait_bound<F: Fn() -> &mut i32>(_f: F) {}
//~^ ERROR: missing lifetime specifier

fn nested_result(_f: impl Fn() -> Result<&mut i32, ()>) {}
//~^ ERROR: missing lifetime specifier

struct Holder<F: Fn() -> &mut i32> {
    //~^ ERROR: missing lifetime specifier
    f: F,
}

fn main() {}
