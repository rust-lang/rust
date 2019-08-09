// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

trait Trait : Sized {
    fn foo(self) -> Self { self }
}

impl Trait for u32 {
    fn foo(self) -> u32 { self }
}

impl Trait for char {
}

fn take_foo_once<T, F: FnOnce(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo<T, F: Fn(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo_mut<T, F: FnMut(T) -> T>(mut f: F, arg: T) -> T {
    (f)(arg)
}

//~ MONO_ITEM fn trait_method_as_argument::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn trait_method_as_argument::take_foo_once[0]<u32, fn(u32) -> u32>
    //~ MONO_ITEM fn trait_method_as_argument::{{impl}}[0]::foo[0]
    //~ MONO_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(u32) -> u32, (u32)>
    take_foo_once(Trait::foo, 0u32);

    //~ MONO_ITEM fn trait_method_as_argument::take_foo_once[0]<char, fn(char) -> char>
    //~ MONO_ITEM fn trait_method_as_argument::Trait[0]::foo[0]<char>
    //~ MONO_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(char) -> char, (char)>
    take_foo_once(Trait::foo, 'c');

    //~ MONO_ITEM fn trait_method_as_argument::take_foo[0]<u32, fn(u32) -> u32>
    //~ MONO_ITEM fn core::ops[0]::function[0]::Fn[0]::call[0]<fn(u32) -> u32, (u32)>
    take_foo(Trait::foo, 0u32);

    //~ MONO_ITEM fn trait_method_as_argument::take_foo[0]<char, fn(char) -> char>
    //~ MONO_ITEM fn core::ops[0]::function[0]::Fn[0]::call[0]<fn(char) -> char, (char)>
    take_foo(Trait::foo, 'c');

    //~ MONO_ITEM fn trait_method_as_argument::take_foo_mut[0]<u32, fn(u32) -> u32>
    //~ MONO_ITEM fn core::ops[0]::function[0]::FnMut[0]::call_mut[0]<fn(char) -> char, (char)>
    take_foo_mut(Trait::foo, 0u32);

    //~ MONO_ITEM fn trait_method_as_argument::take_foo_mut[0]<char, fn(char) -> char>
    //~ MONO_ITEM fn core::ops[0]::function[0]::FnMut[0]::call_mut[0]<fn(u32) -> u32, (u32)>
    take_foo_mut(Trait::foo, 'c');

    0
}
