// https://github.com/rust-lang/rust/issues/4734
//@ run-pass
#![allow(dead_code)]
// Ensures that destructors are run for expressions of the form "e;" where
// `e` is a type which requires a destructor.

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]
#![allow(path_statements)]

struct A { n: isize }
struct B;

static mut NUM_DROPS: usize = 0;

impl Drop for A {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}

impl Drop for B {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}

fn main() {
    assert_eq!(unsafe { NUM_DROPS }, 0);
    { let _a = A { n: 1 }; }
    assert_eq!(unsafe { NUM_DROPS }, 1);
    { A { n: 3 }; }
    assert_eq!(unsafe { NUM_DROPS }, 2);

    { let _b = B; }
    assert_eq!(unsafe { NUM_DROPS }, 3);
    { B; }
    assert_eq!(unsafe { NUM_DROPS }, 4);
}
