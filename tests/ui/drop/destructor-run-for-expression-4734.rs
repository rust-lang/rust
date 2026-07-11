// https://github.com/rust-lang/rust/issues/4734
//@ run-pass
#![allow(dead_code)]
// Ensures that destructors are run for expressions of the form "e;" where
// `e` is a type which requires a destructor.

#![allow(path_statements)]

struct A { n: isize }
struct B;

static mut NUM_DROPS: usize = 0;

fn increment_num_drops() {
    unsafe {
        let num_drops = &raw mut NUM_DROPS;
        num_drops.write(num_drops.read() + 1);
    }
}

fn num_drops() -> usize {
    unsafe { (&raw const NUM_DROPS).read() }
}

impl Drop for A {
    fn drop(&mut self) {
        increment_num_drops();
    }
}

impl Drop for B {
    fn drop(&mut self) {
        increment_num_drops();
    }
}

fn main() {
    assert_eq!(num_drops(), 0);
    { let _a = A { n: 1 }; }
    assert_eq!(num_drops(), 1);
    { A { n: 3 }; }
    assert_eq!(num_drops(), 2);

    { let _b = B; }
    assert_eq!(num_drops(), 3);
    { B; }
    assert_eq!(num_drops(), 4);
}
