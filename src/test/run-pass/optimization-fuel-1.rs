#![crate_name="foo"]

use std::mem::size_of;

// (#55495: The --error-format is to sidestep an issue in our test harness)
// compile-flags: --error-format human -Z fuel=foo=1

struct S1(u8, u16, u8);
struct S2(u8, u16, u8);

fn main() {
    let optimized = (size_of::<S1>() == 4) as usize
        +(size_of::<S2>() == 4) as usize;
    assert_eq!(optimized, 1);
}
