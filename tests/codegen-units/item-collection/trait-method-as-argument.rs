//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![crate_type = "lib"]

trait Trait: Sized {
    fn foo(self) -> Self {
        self
    }
}

impl Trait for u32 {
    fn foo(self) -> u32 {
        self
    }
}

impl Trait for char {}

fn take_foo_once<T, F: FnOnce(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo<T, F: Fn(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo_mut<T, F: FnMut(T) -> T>(mut f: F, arg: T) -> T {
    (f)(arg)
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn take_foo_once::<u32, fn(u32) -> u32 {<u32 as Trait>::foo}>
    //~ MONO_ITEM fn <u32 as Trait>::foo
    //~ MONO_ITEM fn <fn(u32) -> u32 {<u32 as Trait>::foo} as std::ops::FnOnce<(u32,)>>::call_once - shim(fn(u32) -> u32 {<u32 as Trait>::foo})
    take_foo_once(Trait::foo, 0u32);

    //~ MONO_ITEM fn take_foo_once::<char, fn(char) -> char {<char as Trait>::foo}>
    //~ MONO_ITEM fn <char as Trait>::foo
    //~ MONO_ITEM fn <fn(char) -> char {<char as Trait>::foo} as std::ops::FnOnce<(char,)>>::call_once - shim(fn(char) -> char {<char as Trait>::foo})
    take_foo_once(Trait::foo, 'c');

    //~ MONO_ITEM fn take_foo::<u32, fn(u32) -> u32 {<u32 as Trait>::foo}>
    //~ MONO_ITEM fn <fn(u32) -> u32 {<u32 as Trait>::foo} as std::ops::Fn<(u32,)>>::call - shim(fn(u32) -> u32 {<u32 as Trait>::foo})
    take_foo(Trait::foo, 0u32);

    //~ MONO_ITEM fn take_foo::<char, fn(char) -> char {<char as Trait>::foo}>
    //~ MONO_ITEM fn <fn(char) -> char {<char as Trait>::foo} as std::ops::Fn<(char,)>>::call - shim(fn(char) -> char {<char as Trait>::foo})
    take_foo(Trait::foo, 'c');

    //~ MONO_ITEM fn take_foo_mut::<u32, fn(u32) -> u32 {<u32 as Trait>::foo}>
    //~ MONO_ITEM fn <fn(u32) -> u32 {<u32 as Trait>::foo} as std::ops::FnMut<(u32,)>>::call_mut - shim(fn(u32) -> u32 {<u32 as Trait>::foo})
    take_foo_mut(Trait::foo, 0u32);

    //~ MONO_ITEM fn take_foo_mut::<char, fn(char) -> char {<char as Trait>::foo}>
    //~ MONO_ITEM fn <fn(char) -> char {<char as Trait>::foo} as std::ops::FnMut<(char,)>>::call_mut - shim(fn(char) -> char {<char as Trait>::foo})
    take_foo_mut(Trait::foo, 'c');

    0
}
