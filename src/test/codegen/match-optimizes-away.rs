//
// no-system-llvm
// compile-flags: -O
#![crate_type="lib"]

pub enum Three { A, B, C }

pub enum Four { A, B, C, D }

#[no_mangle]
pub fn three_valued(x: Three) -> Three {
    // CHECK-LABEL: @three_valued
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %0
    match x {
        Three::A => Three::A,
        Three::B => Three::B,
        Three::C => Three::C,
    }
}

#[no_mangle]
pub fn four_valued(x: Four) -> Four {
    // CHECK-LABEL: @four_valued
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %0
    match x {
        Four::A => Four::A,
        Four::B => Four::B,
        Four::C => Four::C,
        Four::D => Four::D,
    }
}
