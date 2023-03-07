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

fn main() {
    basic_raw();
}
