// FIXME: this miscompiles with optimizations, see <https://github.com/rust-lang/rust/issues/132898>.
//@compile-flags: -Zmir-opt-level=0

// Test various stacked-borrows-specific things
// (i.e., these do not work the same under TB).
fn main() {
    mut_raw_mut2();
    // direct_mut_to_const_raw();
    two_phase_aliasing_violation();
}

// A variant of `mut_raw_mut` that does *not* get accepted by Tree Borrows.
// It's kind of an accident that we accept it in Stacked Borrows...
fn mut_raw_mut2() {
    unsafe {
        let mut root = 0;
        let to = &mut root as *mut i32;
        *to = 0;
        let _val = root;
        *to = 0;
    }
}

// Make sure that coercing &mut T to *const T produces a writeable pointer.
// TODO: This is currently disabled, waiting on a decision on <https://github.com/rust-lang/rust/issues/56604>
/*fn direct_mut_to_const_raw() {
    let x = &mut 0;
    let y: *const i32 = x;
    unsafe { *(y as *mut i32) = 1; }
    assert_eq!(*x, 1);
}*/

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
