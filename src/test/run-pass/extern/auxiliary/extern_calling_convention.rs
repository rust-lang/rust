// Make sure Rust generates the correct calling convention for extern
// functions.

#[inline(never)]
#[cfg(target_arch = "x86_64")]
pub extern "win64" fn foo(a: isize, b: isize, c: isize, d: isize) {
    assert_eq!(a, 1);
    assert_eq!(b, 2);
    assert_eq!(c, 3);
    assert_eq!(d, 4);

    println!("a: {}, b: {}, c: {}, d: {}",
             a, b, c, d)
}

#[inline(never)]
#[cfg(not(target_arch = "x86_64"))]
pub extern fn foo(a: isize, b: isize, c: isize, d: isize) {
    assert_eq!(a, 1);
    assert_eq!(b, 2);
    assert_eq!(c, 3);
    assert_eq!(d, 4);

    println!("a: {}, b: {}, c: {}, d: {}",
             a, b, c, d)
}
