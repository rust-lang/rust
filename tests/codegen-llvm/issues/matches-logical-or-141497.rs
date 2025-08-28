// Tests that `matches!` optimizes the same as
// `f == FrameType::Inter || f == FrameType::Switch`.

//@ compile-flags: -Copt-level=3
//@ min-llvm-version: 21

#![crate_type = "lib"]

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key = 0,
    Inter = 1,
    Intra = 2,
    Switch = 3,
}

// CHECK-LABEL: @is_inter_or_switch
#[no_mangle]
pub fn is_inter_or_switch(f: FrameType) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: and i8
    // CHECK-NEXT: icmp
    // CHECK-NEXT: ret
    matches!(f, FrameType::Inter | FrameType::Switch)
}
