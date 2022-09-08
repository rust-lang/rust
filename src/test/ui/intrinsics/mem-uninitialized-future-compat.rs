#![feature(never_type)]
#![deny(mem_uninitialized)]
#![allow(deprecated, invalid_value, dead_code)]

use std::mem::MaybeUninit;

struct UninitStruct {
    a: u32,
    b: char,
}

enum OneVariant {
    Hello,
}

enum TwoVariant {
    Hello,
    Goodbye,
}

enum OneVariantWith<T> {
    Hello(T),
}

unsafe fn unknown_type<T, const N: usize>() {
    std::mem::uninitialized::<T>();
    //~^ ERROR the type `T` is generic, and might not permit being left uninitialized

    std::mem::uninitialized::<[T; N]>();
    //~^ ERROR the type `[T; N]` is generic, and might not permit being left uninitialized

    std::mem::uninitialized::<[char; N]>();
    //~^ ERROR the type `[char; N]` is generic, and might not permit being left uninitialized

    std::mem::uninitialized::<[UninitStruct; N]>();
    //~^ ERROR the type `[UninitStruct; N]` is generic, and might not permit being left uninitialized

    std::mem::uninitialized::<Result<T, !>>();
    //~^ ERROR the type `std::result::Result<T, !>` does not permit being left uninitialized

    std::mem::uninitialized::<OneVariantWith<T>>();
    //~^ ERROR the type `OneVariantWith<T>` is generic, and might not permit being left uninitialized

    std::mem::uninitialized::<[T; 0]>();
    std::mem::uninitialized::<[char; 0]>();
}

fn main() {
    unsafe {
        std::mem::uninitialized::<&'static u32>();
        //~^ ERROR the type `&u32` does not permit being left uninitialized

        std::mem::uninitialized::<Box<u32>>();
        //~^ ERROR the type `std::boxed::Box<u32>` does not permit being left uninitialized

        std::mem::uninitialized::<fn()>();
        //~^ ERROR the type `fn()` does not permit being left uninitialized

        std::mem::uninitialized::<!>();
        //~^ ERROR the type `!` does not permit being left uninitialized

        std::mem::uninitialized::<*mut dyn std::io::Write>();
        //~^ ERROR the type `*mut dyn std::io::Write` does not permit being left uninitialized

        std::mem::uninitialized::<bool>();
        //~^ ERROR the type `bool` does not permit being left uninitialized

        std::mem::uninitialized::<char>();
        //~^ ERROR the type `char` does not permit being left uninitialized

        std::mem::uninitialized::<UninitStruct>();
        //~^ ERROR the type `UninitStruct` does not permit being left uninitialized

        std::mem::uninitialized::<[UninitStruct; 16]>();
        //~^ ERROR the type `[UninitStruct; 16]` does not permit being left uninitialized

        std::mem::uninitialized::<(u32, char)>();
        //~^ ERROR the type `(u32, char)` does not permit being left uninitialized

        std::mem::uninitialized::<TwoVariant>();
        //~^ ERROR the type `TwoVariant` does not permit being left uninitialized

        std::mem::uninitialized::<Result<!, !>>();
        //~^ ERROR the type `std::result::Result<!, !>` does not permit being left uninitialized

        std::mem::uninitialized::<Result<!, u32>>();
        //~^ ERROR the type `std::result::Result<!, u32>` does not permit being left uninitialized

        std::mem::uninitialized::<Option<!>>();
        //~^ ERROR the type `std::option::Option<!>` does not permit being left uninitialized

        std::mem::uninitialized::<OneVariantWith<char>>();
        //~^ ERROR the type `OneVariantWith<char>` does not permit being left uninitialized

        std::mem::uninitialized::<OneVariantWith<!>>();
        //~^ ERROR the type `OneVariantWith<!>` does not permit being left uninitialized

        std::mem::uninitialized::<MaybeUninit<Box<u32>>>();
        std::mem::uninitialized::<usize>();
        std::mem::uninitialized::<f32>();
        std::mem::uninitialized::<*const u8>();
        std::mem::uninitialized::<[u8; 64]>();
        std::mem::uninitialized::<OneVariant>();
        std::mem::uninitialized::<OneVariantWith<u32>>();
    }
}
