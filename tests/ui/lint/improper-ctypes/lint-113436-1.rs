#![deny(improper_ctypes_definitions)]

#[repr(C)]
pub struct Foo {
    a: u8,
    b: (),
}

extern "C" fn foo(x: Foo) -> Foo {
    todo!()
}

struct NotSafe(u32);

#[repr(C)]
pub struct Bar {
    a: u8,
    b: (),
    c: NotSafe,
}

extern "C" fn bar(x: Bar) -> Bar {
    //~^ ERROR `extern` fn uses type `NotSafe`, which is not FFI-safe
    //~^^ ERROR `extern` fn uses type `NotSafe`, which is not FFI-safe
    todo!()
}

fn main() {}
