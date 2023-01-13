// run-pass
// Test that astconv doesn't forget about mutability of &mut str

// pretty-expanded FIXME #23616

fn main() {
    fn foo<T: ?Sized>(_: &mut T) {}
    let _f: fn(&mut str) = foo;
}
