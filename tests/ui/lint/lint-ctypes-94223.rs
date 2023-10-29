#![crate_type = "lib"]
#![deny(improper_ctypes_definitions)]

// Test is split in two files since some warnings arise during WF-checking
// and some later, so we can't get them all in the same file.

type Foo = extern "C" fn([u8]);
//~^ ERROR `extern` fn uses type `[u8]`, which is not FFI-safe

pub trait FooTrait {
    type FooType;
}

pub type Foo2<T> = extern "C" fn(Option<&<T as FooTrait>::FooType>);
//~^ ERROR `extern` fn uses type `Option<&<T as FooTrait>::FooType>`, which is not FFI-safe

pub struct FfiUnsafe;

#[allow(improper_ctypes_definitions)]
extern "C" fn f(_: FfiUnsafe) {
    unimplemented!()
}

pub static BAD: extern "C" fn(FfiUnsafe) = f;
//~^ ERROR `extern` fn uses type `FfiUnsafe`, which is not FFI-safe

pub static BAD_TWICE: Result<extern "C" fn(FfiUnsafe), extern "C" fn(FfiUnsafe)> = Ok(f);
//~^ ERROR `extern` fn uses type `FfiUnsafe`, which is not FFI-safe
//~^^ ERROR `extern` fn uses type `FfiUnsafe`, which is not FFI-safe

pub const BAD_CONST: extern "C" fn(FfiUnsafe) = f;
//~^ ERROR `extern` fn uses type `FfiUnsafe`, which is not FFI-safe
