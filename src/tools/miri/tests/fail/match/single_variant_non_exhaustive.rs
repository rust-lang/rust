// Like single_variant.rs, but with a non_exhaustive enum, as the generated MIR used to differ
// between these cases.
//
// See: rust-lang/rust#147722
//
// This UB should be detected even with validation disabled.
//@compile-flags: -Zmiri-disable-validation
#![allow(dead_code)]

#[repr(u8)]
#[non_exhaustive]
enum NonExhaustive {
    A(u8) = 42,
}

fn main() {
    unsafe {
        let x: &[u8; 2] = &[21, 37];
        let y: &NonExhaustive = std::mem::transmute(x);
        match y {
            //~^ ERROR: enum value has invalid tag
            NonExhaustive::A(_) => {}
        }
    }
}
