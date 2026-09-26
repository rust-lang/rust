//! Check that LLVM can recognize that all discriminants are powers of two.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/162513>.
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

#![crate_type = "lib"]

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Vsew {
    E8 = 8,
    E16 = 16,
    E32 = 32,
    E64 = 64,
}

#[unsafe(no_mangle)]
pub fn is_power_of_two_match(sew: Vsew) -> u8 {
    // CHECK-LABEL: define{{.*}} i8 @is_power_of_two_match(i8{{.*}} %sew)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %0 = {{.*}}call{{.*}} i8 @llvm.ctpop.i8(i8 %sew)
    // CHECK-NEXT: %1 = icmp eq i8 %0, 1
    // CHECK-NEXT: call void @llvm.assume(i1 %1)
    // CHECK-NEXT: ret i8 %sew
    let res: u8 = match sew {
        Vsew::E8 => 8,
        Vsew::E16 => 16,
        Vsew::E32 => 32,
        Vsew::E64 => 64,
    };
    assert!(res.is_power_of_two());
    res
}

#[unsafe(no_mangle)]
pub fn is_power_of_two_cast(sew: Vsew) -> u8 {
    // CHECK-LABEL: define{{.*}} i8 @is_power_of_two_cast(i8{{.*}} %sew)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %0 = {{.*}}call{{.*}} i8 @llvm.ctpop.i8(i8 %sew)
    // CHECK-NEXT: %1 = icmp eq i8 %0, 1
    // CHECK-NEXT: call void @llvm.assume(i1 %1)
    // CHECK-NEXT: ret i8 %sew
    let res = sew as u8;
    assert!(res.is_power_of_two());
    res
}

impl Vsew {
    #[inline(always)]
    pub const fn bits_width(self) -> u8 {
        match self {
            Self::E8 => 8,
            Self::E16 => 16,
            Self::E32 => 32,
            Self::E64 => 64,
        }
    }
}

#[unsafe(no_mangle)]
pub fn vlmax(sew: Vsew) -> u32 {
    // CHECK-LABEL: define{{.*}} i32 @vlmax(i8 {{.*}}%sew)
    // CHECK-NOT: udiv
    512 / u32::from(sew.bits_width())
}
