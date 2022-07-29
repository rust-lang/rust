//
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

//~ MONO_ITEM fn start
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn take_foo_once::<u32, [fn item {<u32 as Trait>::foo}: fn(u32) -> u32]>
    //~ MONO_ITEM fn <u32 as Trait>::foo
    //~ MONO_ITEM fn <[fn item {<u32 as Trait>::foo}: fn(u32) -> u32] as std::ops::FnOnce<(u32,)>>::call_once - shim([fn item {<u32 as Trait>::foo}: fn(u32) -> u32])
    take_foo_once(Trait::foo, 0u32);

    //~ MONO_ITEM fn take_foo_once::<char, [fn item {<char as Trait>::foo}: fn(char) -> char]>
    //~ MONO_ITEM fn <char as Trait>::foo
    //~ MONO_ITEM fn <[fn item {<char as Trait>::foo}: fn(char) -> char] as std::ops::FnOnce<(char,)>>::call_once - shim([fn item {<char as Trait>::foo}: fn(char) -> char])
    take_foo_once(Trait::foo, 'c');

    //~ MONO_ITEM fn take_foo::<u32, [fn item {<u32 as Trait>::foo}: fn(u32) -> u32]>
    //~ MONO_ITEM fn <[fn item {<u32 as Trait>::foo}: fn(u32) -> u32] as std::ops::Fn<(u32,)>>::call - shim([fn item {<u32 as Trait>::foo}: fn(u32) -> u32])
    take_foo(Trait::foo, 0u32);

    //~ MONO_ITEM fn take_foo::<char, [fn item {<char as Trait>::foo}: fn(char) -> char]>
    //~ MONO_ITEM fn <[fn item {<char as Trait>::foo}: fn(char) -> char] as std::ops::Fn<(char,)>>::call - shim([fn item {<char as Trait>::foo}: fn(char) -> char])
    take_foo(Trait::foo, 'c');

    //~ MONO_ITEM fn take_foo_mut::<u32, [fn item {<u32 as Trait>::foo}: fn(u32) -> u32]>
    //~ MONO_ITEM fn <[fn item {<u32 as Trait>::foo}: fn(u32) -> u32] as std::ops::FnMut<(u32,)>>::call_mut - shim([fn item {<u32 as Trait>::foo}: fn(u32) -> u32])
    take_foo_mut(Trait::foo, 0u32);

    //~ MONO_ITEM fn take_foo_mut::<char, [fn item {<char as Trait>::foo}: fn(char) -> char]>
    //~ MONO_ITEM fn <[fn item {<char as Trait>::foo}: fn(char) -> char] as std::ops::FnMut<(char,)>>::call_mut - shim([fn item {<char as Trait>::foo}: fn(char) -> char])
    take_foo_mut(Trait::foo, 'c');

    0
}
