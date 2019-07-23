// This returns a miri pointer at type usize, if the argument is a proper pointer
fn transmute_ptr_to_int<T>(x: *const T) -> usize {
    unsafe { std::mem::transmute(x) }
}

fn main() {
    // Some casting-to-int with arithmetic.
    let x = &42 as *const i32 as usize;
    let y = x * 2;
    assert_eq!(y, x + x);
    let z = y as u8 as usize;
    assert_eq!(z, y % 256);

    // Pointer string formatting! We can't check the output as it changes when libstd changes,
    // but we can make sure Miri does not error.
    format!("{:?}", &mut 13 as *mut _);

    // Check that intptrcast is triggered for explicit casts and that it is consistent with
    // transmuting.
    let a: *const i32 = &42;
    let b = transmute_ptr_to_int(a) as u8;
    let c = a as usize as u8;
    assert_eq!(b, c);
}
