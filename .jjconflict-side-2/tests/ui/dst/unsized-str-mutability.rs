//! regression test for <https://github.com/rust-lang/rust/issues/17361>
//! Test that HIR ty lowering doesn't forget about mutability of `&mut str`.
//@ run-pass

fn main() {
    fn foo<T: ?Sized>(_: &mut T) {}
    let _f: fn(&mut str) = foo;
}
