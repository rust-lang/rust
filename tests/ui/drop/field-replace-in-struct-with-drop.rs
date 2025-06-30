//! Test that field replacement in structs with destructors doesn't trigger warnings.
//!
//! Back in 2016, the compiler would incorrectly warn about "moving out of type with dtor"
//! when you assigned to a field of a struct that has a Drop impl. But this is perfectly
//! fine - we're replacing the field, not moving out of it. The old value gets dropped
//! and the new value takes its place.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/34101>.

//@ check-pass

struct Foo(String);

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn test_inline_replacement() {
    // dummy variable so `f` gets assigned `var1` in MIR for both functions
    let _s = ();
    let mut f = Foo(String::from("foo"));
    f.0 = String::from("bar"); // This should not warn
}

fn test_outline_replacement() {
    let _s = String::from("foo");
    let mut f = Foo(_s);
    f.0 = String::from("bar"); // This should not warn either
}

fn main() {
    test_inline_replacement();
    test_outline_replacement();
}
