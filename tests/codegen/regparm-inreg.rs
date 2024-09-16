// Checks how `regparm` flag works with different calling conventions:
// marks function arguments as "inreg" like the C/C++ compilers for the platforms.
// x86 only.

//@ compile-flags: --target i686-unknown-linux-gnu -O -C no-prepopulate-passes
//@ needs-llvm-components: x86

//@ revisions:regpram0 regpram1 regpram2 regpram3 regpram4
//@[regpram0] compile-flags: -Zregparm=0
//@[regpram1] compile-flags: -Zregparm=1
//@[regpram2] compile-flags: -Zregparm=2
//@[regpram3] compile-flags: -Zregparm=3
//@[regpram4] compile-flags: -Zregparm=4

#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub mod tests {
    // regparm doesn't work for "fastcall" calling conv (only 2 inregs)
    // CHECK: @f1(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "fastcall" fn f1(_: i32, _: i32, _: i32) {}

    // regpram0: @f2(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram1: @f2(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram2: @f2(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regpram3: @f2(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regpram4: @f2(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "Rust" fn f2(_: i32, _: i32, _: i32) {}

    // regpram0: @f3(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram1: @f3(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram2: @f3(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regpram3: @f3(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regpram4: @f3(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "C" fn f3(_: i32, _: i32, _: i32) {}

    // regpram0: @f4(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram1: @f4(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram2: @f4(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regpram3: @f4(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regpram4: @f4(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "cdecl" fn f4(_: i32, _: i32, _: i32) {}

    // regpram0: @f5(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram1: @f5(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regpram2: @f5(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regpram3: @f5(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    // regpram4: @f5(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "stdcall" fn f5(_: i32, _: i32, _: i32) {}

    // regparm doesn't work for thiscall
    // CHECK: @f6(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "thiscall" fn f6(_: i32, _: i32, _: i32) {}

    struct s1 {
        x1: i32,
    }
    // regpram0: @f7(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3, i32 noundef %_4)
    // regpram1: @f7(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3, i32 noundef %_4)
    // regpram2: @f7(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3, i32 noundef %_4)
    // regpram3: @f7(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3,
    // regpram3: i32 noundef %_4)
    // regpram4: @f7(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3,
    // regpram4: i32 inreg noundef %_4)
    #[no_mangle]
    pub extern "C" fn f7(_: i32, _: i32, _: s1, _: i32) {}

    #[repr(C)]
    struct s2 {
        x1: i32,
        x2: i32,
    }
    // regpram0: @f8(i32 noundef %_1, i32 noundef %_2, ptr {{.*}} %_3, i32 noundef %_4)
    // regpram1: @f8(i32 inreg noundef %_1, i32 noundef %_2, ptr {{.*}} %_3, i32 noundef %_4)
    // regpram2: @f8(i32 inreg noundef %_1, i32 inreg noundef %_2, ptr {{.*}} %_3, i32 noundef %_4)
    // regpram3: @f8(i32 inreg noundef %_1, i32 inreg noundef %_2, ptr {{.*}} %_3,
    // regpram3: i32 inreg noundef %_4)
    // regpram4: @f8(i32 inreg noundef %_1, i32 inreg noundef %_2, ptr {{.*}} %_3,
    // regpram4: i32 inreg noundef %_4)
    #[no_mangle]
    pub extern "C" fn f8(_: i32, _: i32, _: s2, _: i32) {}
}
