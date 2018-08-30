#![feature(extern_types)]

extern {
    type Foo;
}

fn main() {
    let x: &Foo = unsafe { &*(16 as *const Foo) };
    let _y: &Foo = &*x;
}
