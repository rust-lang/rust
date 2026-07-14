// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. The discriminator values were obtained
// from Clang by compiling equivalent C code (included). Both compilers must generate identical
// values.
//
// Make sure that rust only uses the final part of struct's name (`Foo` or `Bar`), so that the
// discriminators are `F3FooE` and `F3BarE`, not using def path for the base of encoding.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;
use minicore::hint::black_box;

#[repr(C)]
pub struct Foo {
    x: i32,
}

#[repr(C)]
pub struct Bar {
    x: i32,
}

extern "C" fn takes_foo(_: Foo) {}
extern "C" fn takes_bar(_: Bar) {}

#[used]
// DISC-DAG: @{{.*}}FOO_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes_foo, i32 0, i64 58649)
// Without type discriminators all the functions are the same, so compiler is able to use both
// takes_foo/take_bar.
// NO_DISC-DAG: @{{.*}}FOO_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes_{{foo|bar}}, i32 0)
static FOO_FNPTR: extern "C" fn(Foo) = takes_foo;

#[used]
// DISC-DAG: @{{.*}}BAR_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes_bar, i32 0, i64 41614)
// NO_DISC-DAG: @{{.*}}BAR_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes_{{foo|bar}}, i32 0)
static BAR_FNPTR: extern "C" fn(Bar) = takes_bar;

// While not possible to express in C, we could force it through C++ path with something along the
// lines of:
// ```c++
// namespace a {
// struct SameName {
//   int x;
// };
//
// void takes(SameName) {}
// }
//
// namespace b {
// struct SameName {
//   int x;
// };
//
// void takes(SameName) {}
// }
//
// void (*a_fnptr)(a::SameName) = a::takes;
// void (*b_fnptr)(b::SameName) = b::takes;
// ```
// Make sure that Rust uses `Fv8SameNameE` for both `A_FNPTR` and `B_FNPTR`, not
// `Fv11a::SameNameE`, or `Fv11b::SameNameE`.

mod a {
    #[repr(C)]
    pub struct SameName {
        pub x: i32,
    }

    pub extern "C" fn takes(_: SameName) {}
}

mod b {
    #[repr(C)]
    pub struct SameName {
        pub x: i32,
    }

    pub extern "C" fn takes(_: SameName) {}
}

#[used]
// DISC-DAG: @{{.*}}A_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes, i32 0, i64 57535)
// NO_DISC-DAG: @{{.*}}A_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes{{.*}}, i32 0)
static A_FNPTR: extern "C" fn(a::SameName) = a::takes;

#[used]
// DISC-DAG: @{{.*}}B_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes, i32 0, i64 57535)
// NO_DISC-DAG: @{{.*}}B_FNPTR = constant ptr ptrauth (ptr @{{.*}}takes{{.*}}, i32 0)
static B_FNPTR: extern "C" fn(b::SameName) = b::takes;
