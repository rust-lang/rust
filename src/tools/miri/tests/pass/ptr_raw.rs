fn basic_raw() {
    let mut x = 12;
    let x = &mut x;

    assert_eq!(*x, 12);

    let raw = x as *mut i32;
    unsafe {
        *raw = 42;
    }

    assert_eq!(*x, 42);

    let raw = x as *mut i32;
    unsafe {
        *raw = 12;
    }
    *x = 23;

    assert_eq!(*x, 23);
}

fn assign_overlapping() {
    // Test an assignment where LHS and RHS alias.
    // In Mir, that's UB (see `fail/overlapping_assignment.rs`), but in surface Rust this is allowed.
    let mut mem = [0u32; 4];
    let ptr = &mut mem as *mut [u32; 4];
    unsafe { *ptr = *ptr };
}

fn main() {
    basic_raw();
    assign_overlapping();
}
