// run-pass
#![allow(const_err)]

// compile-flags: -O

fn main() {
    let x = &(0u32 - 1);
    assert_eq!(*x, u32::max_value())
}
