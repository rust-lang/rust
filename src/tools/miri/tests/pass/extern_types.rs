#![feature(extern_types)]

extern "C" {
    type Foo;
}

fn main() {
    let x: &Foo = unsafe { &*(16 as *const Foo) };
    let _y: &Foo = &*x;
}
