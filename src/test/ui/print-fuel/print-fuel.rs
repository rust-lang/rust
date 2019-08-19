#![crate_name="foo"]
#![allow(dead_code)]

// (#55495: The --error-format is to sidestep an issue in our test harness)
// compile-flags: --error-format human -Z print-fuel=foo
// build-pass (FIXME(62277): could be check-pass?)

struct S1(u8, u16, u8);
struct S2(u8, u16, u8);
struct S3(u8, u16, u8);

fn main() {
}
