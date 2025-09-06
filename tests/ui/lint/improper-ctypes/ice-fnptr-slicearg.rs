#![crate_type = "lib"]
#![deny(improper_c_fn_definitions, improper_c_callbacks)]

// Issue: https://github.com/rust-lang/rust/issues/94223
// ice when a FnPtr has an unsized array argument

pub fn bad(f: extern "C" fn([u8])) {}
//~^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe

pub fn bad_twice(f: Result<extern "C" fn([u8]), extern "C" fn([u8])>) {}
//~^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe
//~^^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe

struct BadStruct(extern "C" fn([u8]));
//~^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe

enum BadEnum {
    A(extern "C" fn([u8])),
    //~^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe
}

enum BadUnion {
    A(extern "C" fn([u8])),
    //~^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe
}

type Foo = extern "C" fn([u8]);
//~^ ERROR `extern` callback uses type `[u8]`, which is not FFI-safe

pub trait FooTrait {
    type FooType;
}

pub type Foo2<T> = extern "C" fn(Option<&<T as FooTrait>::FooType>);
//~^ ERROR `extern` callback uses type `Option<&<T as FooTrait>::FooType>`, which is not FFI-safe

pub struct FfiUnsafe;

#[allow(improper_c_fn_definitions)]
extern "C" fn f(_: FfiUnsafe) {
    unimplemented!()
}

pub static BAD: extern "C" fn(FfiUnsafe) = f;
//~^ ERROR `extern` callback uses type `FfiUnsafe`, which is not FFI-safe

pub static BAD_TWICE: Result<extern "C" fn(FfiUnsafe), extern "C" fn(FfiUnsafe)> = Ok(f);
//~^ ERROR `extern` callback uses type `FfiUnsafe`, which is not FFI-safe
//~^^ ERROR `extern` callback uses type `FfiUnsafe`, which is not FFI-safe

pub const BAD_CONST: extern "C" fn(FfiUnsafe) = f;
//~^ ERROR `extern` callback uses type `FfiUnsafe`, which is not FFI-safe
