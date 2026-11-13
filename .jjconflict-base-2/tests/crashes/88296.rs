//@ known-bug: #88296

#![feature(specialization)]

trait Foo {
    type Bar;
}

impl<T> Foo for T {
    default type Bar = u32;
}

impl Foo for i32 {
    type Bar = i32;
}

extern "C" {
    #[allow(unused)]
    // OK as Foo::Bar is explicitly defined for i32
    static OK: <i32 as Foo>::Bar;

    #[allow(unused)]
    // ICE in the improper_ctypes lint
    //  as Foo::Bar is only default implemented for ()
    static ICE: <() as Foo>::Bar;
}
pub fn main()  {}
