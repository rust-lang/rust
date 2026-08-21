// Ensure const-eval accepts a truncated value when it satisfies the destination
// type's invariants.

//@ run-pass
use std::mem::transmute_copy;
use std::num::NonZeroU8;

const NONZERO_LEAD: (u8, u8) = (5, 0);
const OK: NonZeroU8 = unsafe { transmute_copy::<(u8, u8), NonZeroU8>(&NONZERO_LEAD) };

fn main() {
    assert_eq!(OK.get(), 5);
}
