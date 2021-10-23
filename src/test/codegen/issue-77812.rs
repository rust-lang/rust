// compile-flags: -O
#![crate_type = "lib"]

// Test that LLVM can eliminate the unreachable `Variant::Zero` branch.

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Variant {
    Zero,
    One,
    Two,
}

extern {
    fn exf1();
    fn exf2();
}

pub static mut GLOBAL: Variant = Variant::Zero;

// CHECK-LABEL: @issue_77812
#[no_mangle]
pub unsafe fn issue_77812() {
    let g = GLOBAL;
    if g != Variant::Zero {
        match g {
            Variant::One => exf1(),
            Variant::Two => exf2(),
            // CHECK-NOT: panic
            Variant::Zero => panic!(),
        }
    }
}
