// Checks how `reg-struct-return` flag works with different calling conventions:
// Return struct with 8/16/32/64 bit size will be converted into i8/i16/i32/i64
// (like abi_return_struct_as_int target spec).
// x86 only.

//@ compile-flags: --target i686-unknown-linux-gnu -Zreg-struct-return -O -C no-prepopulate-passes
//@ needs-llvm-components: x86

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[repr(C)]
pub struct Foo {
    x: u32,
    y: u32,
}

pub mod tests {
    use Foo;

    // CHECK: i64 @f1()
    #[no_mangle]
    pub extern "fastcall" fn f1() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // CHECK: i64 @f2()
    #[no_mangle]
    pub extern "Rust" fn f2() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // CHECK: i64 @f3()
    #[no_mangle]
    pub extern "C" fn f3() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // CHECK: i64 @f4()
    #[no_mangle]
    pub extern "cdecl" fn f4() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // CHECK: i64 @f5()
    #[no_mangle]
    pub extern "stdcall" fn f5() -> Foo {
        Foo { x: 1, y: 2 }
    }

    // CHECK: i64 @f6()
    #[no_mangle]
    pub extern "thiscall" fn f6() -> Foo {
        Foo { x: 1, y: 2 }
    }
}
