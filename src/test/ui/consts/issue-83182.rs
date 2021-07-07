// stderr-per-bitwidth

use std::mem;
struct MyStr(str);
const MYSTR_NO_INIT: &MyStr = unsafe { mem::transmute::<&[_], _>(&[&()]) };
//~^ ERROR: it is undefined behavior to use this value
//~| type validation failed at .<deref>.0: encountered a pointer in `str`
fn main() {}
