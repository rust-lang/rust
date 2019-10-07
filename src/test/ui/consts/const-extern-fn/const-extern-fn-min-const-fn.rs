#![feature(const_extern_fn)]

const extern fn unsize(x: &[u8; 3]) -> &[u8] { x }
//~^ ERROR unsizing casts are not allowed in const fn
const unsafe extern "C" fn closure() -> fn() { || {} }
//~^ ERROR function pointers in const fn are unstable
const unsafe extern fn use_float() { 1.0 + 1.0; }
//~^ ERROR only int, `bool` and `char` operations are stable in const fn
const extern "C" fn ptr_cast(val: *const u8) { val as usize; }
//~^ ERROR casting pointers to ints is unstable in const fn


fn main() {}
