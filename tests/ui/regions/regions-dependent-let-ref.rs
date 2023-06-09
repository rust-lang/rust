// run-pass
// Test lifetimes are linked properly when we take reference
// to interior.

// pretty-expanded FIXME #23616

struct Foo(isize);
pub fn main() {
    // Here the lifetime of the `&` should be at least the
    // block, since a ref binding is created to the interior.
    let &Foo(ref _x) = &Foo(3);
}
