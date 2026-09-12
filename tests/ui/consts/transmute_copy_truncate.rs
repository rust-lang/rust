// Ensure const-eval validates the truncated value against the destination
// type's invariants: reject leading zero.
use std::mem::transmute_copy;
use std::num::NonZeroU8;

const ZERO_LEAD: (u8, u8) = (0, 5);
const BAD: NonZeroU8 = unsafe { transmute_copy::<(u8, u8), NonZeroU8>(&ZERO_LEAD) };
//~^ ERROR constructing invalid value of type NonZero<u8>: at .0.0, encountered 0, but expected something greater or equal to 1

fn main() {}
