#![feature(core_intrinsics)]

use std::intrinsics;

const SHR: u8 = unsafe { intrinsics::unchecked_shr(5_u8, 8) };
//~^ ERROR any use of this value will cause an error
const SHL: u8 = unsafe { intrinsics::unchecked_shl(5_u8, 8) };
//~^ ERROR any use of this value will cause an error

fn main() {
}
