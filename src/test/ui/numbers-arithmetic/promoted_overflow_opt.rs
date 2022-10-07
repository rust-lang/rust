// run-pass

// compile-flags: -O

fn main() {
    let x = &(0u32 - 1);
    assert_eq!(*x, u32::MAX)
}
