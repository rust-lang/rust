#![deny(improper_c_fn_definitions)]

// Issue: https://github.com/rust-lang/rust/issues/113436
// `()` in (fnptr!) return types and ADT fields should be safe

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
    //~^ ERROR `extern` fn uses type `Bar`, which is not FFI-safe
    //~^^ ERROR `extern` fn uses type `Bar`, which is not FFI-safe
    todo!()
}

fn main() {}
