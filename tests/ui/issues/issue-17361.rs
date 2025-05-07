//@ run-pass
// Test that HIR ty lowering doesn't forget about mutability of `&mut str`.


fn main() {
    fn foo<T: ?Sized>(_: &mut T) {}
    let _f: fn(&mut str) = foo;
}
