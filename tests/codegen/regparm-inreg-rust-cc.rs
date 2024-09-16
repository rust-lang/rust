// Checks how `regparm` flag works with Rust calling convention with array types.
// When there is a small array type in signature (casted to combined int type),
//   inregs will not be set. PassMode::Cast is unsupported.
// x86 only.

//@ compile-flags: --target i686-unknown-linux-gnu -O -C no-prepopulate-passes
//@ needs-llvm-components: x86

//@ revisions:regparm0 regparm1 regparm2 regparm3
//@[regparm0] compile-flags: -Zregparm=0
//@[regparm1] compile-flags: -Zregparm=1
//@[regparm2] compile-flags: -Zregparm=2
//@[regparm3] compile-flags: -Zregparm=3

#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub mod tests {
    // CHECK: @f1(i16 %0, i32 noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "Rust" fn f1(_: [u8; 2], _: i32, _: i32) {}

    // CHECK: @f2(i24 %0, i32 noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "Rust" fn f2(_: [u8; 3], _: i32, _: i32) {}

    // regparm0: @f3(ptr {{.*}} %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm1: @f3(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regparm2: @f3(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regparm3: @f3(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "Rust" fn f3(_: [u8; 7], _: i32, _: i32) {}

    // regparm0: @f4(ptr {{.*}} %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm1: @f4(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regparm2: @f4(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regparm3: @f4(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "Rust" fn f4(_: [u8; 11], _: i32, _: i32) {}

    // regparm0: @f5(ptr {{.*}} %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm1: @f5(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regparm2: @f5(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regparm3: @f5(ptr {{.*}} %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "Rust" fn f5(_: [u8; 33], _: i32, _: i32) {}
}
