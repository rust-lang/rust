// FIXME: this miscompiles with optimizations, see <https://github.com/rust-lang/rust/issues/132898>.
//@compile-flags: -Zmir-opt-level=0

// Test various stacked-borrows-specific things
// (i.e., these do not work the same under TB).
fn main() {
    two_phase_aliasing_violation();
}

// This one really shouldn't be accepted, but since we treat 2phase as raw, we do accept it.
// Tree Borrows rejects it.
fn two_phase_aliasing_violation() {
    struct Foo(u64);
    impl Foo {
        fn add(&mut self, n: u64) -> u64 {
            self.0 + n
        }
    }

    let mut f = Foo(0);
    let alias = &mut f.0 as *mut u64;
    let res = f.add(unsafe {
        *alias = 42;
        0
    });
    assert_eq!(res, 42);
}
