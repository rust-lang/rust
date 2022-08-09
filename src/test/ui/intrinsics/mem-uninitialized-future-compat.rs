#![feature(never_type)]
#![deny(mem_uninitialized)]
#![allow(deprecated, invalid_value, dead_code)]

use std::mem::MaybeUninit;

struct UninitStruct {
    a: u32,
    b: char,
}

unsafe fn unknown_type<T, const N: usize>() {
    std::mem::uninitialized::<T>();
    //~^ ERROR the type `T` does not definitely permit being left uninitialized

    std::mem::uninitialized::<[T; N]>();
    //~^ ERROR the type `[T; N]` does not definitely permit being left uninitialized

    std::mem::uninitialized::<[char; N]>();
    //~^ ERROR the type `[char; N]` does not definitely permit being left uninitialized

    std::mem::uninitialized::<[T; 0]>();
    std::mem::uninitialized::<[char; 0]>();
}

fn main() {
    unsafe {
        std::mem::uninitialized::<&'static u32>();
        //~^ ERROR the type `&u32` does not definitely permit being left uninitialized

        std::mem::uninitialized::<Box<u32>>();
        //~^ ERROR the type `std::boxed::Box<u32>` does not definitely permit being left uninitialized

        std::mem::uninitialized::<fn()>();
        //~^ ERROR the type `fn()` does not definitely permit being left uninitialized

        std::mem::uninitialized::<!>();
        //~^ ERROR the type `!` does not definitely permit being left uninitialized

        std::mem::uninitialized::<*mut dyn std::io::Write>();
        //~^ ERROR the type `*mut dyn std::io::Write` does not definitely permit being left uninitialized

        std::mem::uninitialized::<bool>();
        //~^ ERROR the type `bool` does not definitely permit being left uninitialized

        std::mem::uninitialized::<char>();
        //~^ ERROR the type `char` does not definitely permit being left uninitialized

        std::mem::uninitialized::<UninitStruct>();
        //~^ ERROR the type `UninitStruct` does not definitely permit being left uninitialized

        std::mem::uninitialized::<(u32, char)>();
        //~^ ERROR the type `(u32, char)` does not definitely permit being left uninitialized

        std::mem::uninitialized::<MaybeUninit<Box<u32>>>();
        std::mem::uninitialized::<usize>();
        std::mem::uninitialized::<f32>();
        std::mem::uninitialized::<*const u8>();
    }
}
