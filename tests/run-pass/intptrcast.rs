// This returns a miri pointer at type usize, if the argument is a proper pointer
fn transmute_ptr_to_int<T>(x: *const T) -> usize {
    unsafe { std::mem::transmute(x) }
}

fn cast() {
    // Some casting-to-int with arithmetic.
    let x = &42 as *const i32 as usize;
    let y = x * 2;
    assert_eq!(y, x + x);
    let z = y as u8 as usize;
    assert_eq!(z, y % 256);
}

/// Test usize->ptr cast for dangling and OOB address.
/// That is safe, and thus has to work.
fn cast_dangling() {
    let b = Box::new(0);
    let x = &*b as *const i32 as usize;
    drop(b);
    let _val = x as *const i32;

    let b = Box::new(0);
    let mut x = &*b as *const i32 as usize;
    x += 0x100;
    let _val = x as *const i32;
}

fn format() {
    // Pointer string formatting! We can't check the output as it changes when libstd changes,
    // but we can make sure Miri does not error.
    format!("{:?}", &mut 13 as *mut _);
}

fn transmute() {
    // Check that intptrcast is triggered for explicit casts and that it is consistent with
    // transmuting.
    let a: *const i32 = &42;
    let b = transmute_ptr_to_int(a) as u8;
    let c = a as usize as u8;
    assert_eq!(b, c);
}

fn ptr_bitops1() {
    let bytes = [0i8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let one = bytes.as_ptr().wrapping_offset(1);
    let three = bytes.as_ptr().wrapping_offset(3);
    let res = (one as usize) | (three as usize);
    format!("{}", res);
}

fn ptr_bitops2() {
    let val = 13usize;
    let addr = &val as *const _ as usize;
    let _val = addr & 13;
}

fn ptr_eq_dangling() {
    let b = Box::new(0);
    let x = &*b as *const i32; // soon-to-be dangling
    drop(b);
    let b = Box::new(0);
    let y = &*b as *const i32; // different allocation
    // They *could* be equal if memory was reused, but probably are not.
    assert!(x != y);
}

fn ptr_eq_out_of_bounds() {
    let b = Box::new(0);
    let x = (&*b as *const i32).wrapping_sub(0x800); // out-of-bounds
    let b = Box::new(0);
    let y = &*b as *const i32; // different allocation
    // They *could* be equal (with the right base addresses), but probably are not.
    assert!(x != y);
}

fn ptr_eq_out_of_bounds_null() {
    let b = Box::new(0);
    let x = (&*b as *const i32).wrapping_sub(0x800); // out-of-bounds
    // This *could* be NULL (with the right base address), but probably is not.
    assert!(x != std::ptr::null());
}

fn ptr_eq_integer() {
    let b = Box::new(0);
    let x = &*b as *const i32;
    // These *could* be equal (with the right base address), but probably are not.
    assert!(x != 64 as *const i32);
}

fn main() {
    cast();
    cast_dangling();
    format();
    transmute();
    ptr_bitops1();
    ptr_bitops2();
    ptr_eq_dangling();
    ptr_eq_out_of_bounds();
    ptr_eq_out_of_bounds_null();
    ptr_eq_integer();
}
