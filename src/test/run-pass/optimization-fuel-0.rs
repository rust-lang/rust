#![crate_name="foo"]

use std::mem::size_of;

// (#55495: The --error-format is to sidestep an issue in our test harness)
// compile-flags: --error-format human -Z fuel=foo=0

struct S1(u8, u16, u8);
struct S2(u8, u16, u8);

fn main() {
    assert_eq!(size_of::<S1>(), 6);
    assert_eq!(size_of::<S2>(), 6);
}
