// run-pass
#![feature(explicit_tail_calls)]
#![feature(c_variadic)]

pub unsafe extern "C" fn c_variadic_f(x: u8, mut args: ...) {
    assert_eq!(x, 12);
    let (a, b) = (args.arg::<u32>(), args.arg::<u32>());
    become c_variadic_g(x + 1, a, b, 3u32);
}

pub unsafe extern "C" fn c_variadic_g(x: u8, mut args: ...) {
    assert_eq!(x, 13);
    assert_eq!(args.arg::<u32>(), 1);
    assert_eq!(args.arg::<u32>(), 2);
    assert_eq!(args.arg::<u32>(), 3);
}

fn main() {
    unsafe { c_variadic_f(12u8, 1u32, 2u32) };
}
