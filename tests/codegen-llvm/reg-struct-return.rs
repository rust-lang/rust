// Checks how `reg-struct-return` flag works with different calling conventions:
// Return struct with 8/16/32/64 bit size will be converted into i8/i16/i32/i64
// (like abi_return_struct_as_int target spec).
// x86 only.

//@ revisions: ENABLED DISABLED
//@ add-core-stubs
//@ compile-flags: --target i686-unknown-linux-gnu -Cno-prepopulate-passes -Copt-level=3
//@ [ENABLED] compile-flags: -Zreg-struct-return
//@ needs-llvm-components: x86

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Foo {
    x: u32,
    y: u32,
}

#[repr(C)]
pub struct Foo1 {
    x: u32,
}

#[repr(C)]
pub struct Foo2 {
    x: bool,
    y: bool,
    z: i16,
}

#[repr(C)]
pub struct Foo3 {
    x: i16,
    y: bool,
    z: bool,
}

#[repr(C)]
pub struct Foo4 {
    x: char,
    y: bool,
    z: u8,
}

#[repr(C)]
pub struct Foo5 {
    x: u32,
    y: u16,
    z: u8,
    a: bool,
}

#[repr(C)]
pub struct FooOversize1 {
    x: u32,
    y: u32,
    z: u32,
}

#[repr(C)]
pub struct FooOversize2 {
    f0: u16,
    f1: u16,
    f2: u16,
    f3: u16,
    f4: u16,
}

#[repr(C)]
pub struct FooFloat1 {
    x: f32,
    y: f32,
}

#[repr(C)]
pub struct FooFloat2 {
    x: f64,
}

#[repr(C)]
pub struct FooFloat3 {
    x: f32,
}

pub mod tests {
    use {
        Foo, Foo1, Foo2, Foo3, Foo4, Foo5, FooFloat1, FooFloat2, FooFloat3, FooOversize1,
        FooOversize2,
    };

    // ENABLED: i64 @f1()
    // DISABLED: void @f1(ptr {{.*}}sret
    #[no_mangle]
    pub extern "fastcall" fn f1() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // CHECK: { i32, i32 } @f2()
    #[no_mangle]
    pub extern "Rust" fn f2() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // ENABLED: i64 @f3()
    // DISABLED: void @f3(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f3() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // ENABLED: i64 @f4()
    // DISABLED: void @f4(ptr {{.*}}sret
    #[no_mangle]
    pub extern "cdecl" fn f4() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // ENABLED: i64 @f5()
    // DISABLED: void @f5(ptr {{.*}}sret
    #[no_mangle]
    pub extern "stdcall" fn f5() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // ENABLED: i64 @f6()
    // DISABLED: void @f6(ptr {{.*}}sret
    #[no_mangle]
    pub extern "thiscall" fn f6() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // ENABLED: i32 @f7()
    // DISABLED: void @f7(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f7() -> Foo1 {
        Foo1 { x: 1 }
    }

    // ENABLED: i32 @f8()
    // DISABLED: void @f8(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f8() -> Foo2 {
        Foo2 { x: true, y: false, z: 5 }
    }

    // ENABLED: i32 @f9()
    // DISABLED: void @f9(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f9() -> Foo3 {
        Foo3 { x: 5, y: false, z: true }
    }

    // ENABLED: i64 @f10()
    // DISABLED: void @f10(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f10() -> Foo4 {
        Foo4 { x: 'x', y: true, z: 170 }
    }

    // ENABLED: i64 @f11()
    // DISABLED: void @f11(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f11() -> Foo5 {
        Foo5 { x: 1, y: 2, z: 3, a: true }
    }

    // CHECK: void @f12(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f12() -> FooOversize1 {
        FooOversize1 { x: 1, y: 2, z: 3 }
    }

    // CHECK: void @f13(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f13() -> FooOversize2 {
        FooOversize2 { f0: 1, f1: 2, f2: 3, f3: 4, f4: 5 }
    }

    // ENABLED: i64 @f14()
    // DISABLED: void @f14(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f14() -> FooFloat1 {
        FooFloat1 { x: 1.0, y: 1.0 }
    }

    // ENABLED: double @f15()
    // DISABLED: void @f15(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f15() -> FooFloat2 {
        FooFloat2 { x: 1.0 }
    }

    // ENABLED: float @f16()
    // DISABLED: void @f16(ptr {{.*}}sret
    #[no_mangle]
    pub extern "C" fn f16() -> FooFloat3 {
        FooFloat3 { x: 1.0 }
    }
}
