//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
#![crate_type = "lib"]

pub enum Three {
    A,
    B,
    C,
}

#[repr(u16)]
pub enum Four {
    A,
    B,
    C,
    D,
}

#[no_mangle]
pub fn three_valued(x: Three) -> Three {
    // CHECK-LABEL: i8 @three_valued(i8{{.+}}%x)
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %x
    match x {
        Three::A => Three::A,
        Three::B => Three::B,
        Three::C => Three::C,
    }
}

#[no_mangle]
pub fn four_valued(x: Four) -> Four {
    // CHECK-LABEL: i16 @four_valued(i16{{.+}}%x)
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i16 %x
    match x {
        Four::A => Four::A,
        Four::B => Four::B,
        Four::C => Four::C,
        Four::D => Four::D,
    }
}
