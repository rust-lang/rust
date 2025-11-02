//@ add-core-stubs
//@ edition: 2018
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(cmse_nonsecure_entry, c_variadic, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[lang = "va_list"]
struct VaList(*mut u8);

unsafe extern "cmse-nonsecure-entry" fn c_variadic(_: u32, _: ...) {
    //~^ ERROR `...` is not supported for `extern "cmse-nonsecure-entry"` functions
}

// A regression test for https://github.com/rust-lang/rust/issues/132142
async unsafe extern "cmse-nonsecure-entry" fn async_and_c_variadic(_: ...) {
    //~^ ERROR `...` is not supported for `extern "cmse-nonsecure-entry"` functions
    //~| ERROR functions cannot be both `async` and C-variadic
}

// Below are the lang items that are required for a program that defines an `async` function.
// Without them, the ICE that is tested for here is not reached for this target. For now they are in
// this file, but they may be moved into `minicore` if/when other `#[no_core]` tests want to use
// them.

// NOTE: in `core` this type uses `NonNull`.
#[lang = "ResumeTy"]
pub struct ResumeTy(*mut Context<'static>);

#[lang = "future_trait"]
pub trait Future {
    /// The type of value produced on completion.
    #[lang = "future_output"]
    type Output;

    // NOTE: misses the `poll` method.
}

#[lang = "async_drop"]
pub trait AsyncDrop {
    // NOTE: misses the `drop` method.
}

#[lang = "Poll"]
pub enum Poll<T> {
    #[lang = "Ready"]
    Ready(T),

    #[lang = "Pending"]
    Pending,
}

#[lang = "Context"]
pub struct Context<'a> {
    // NOTE: misses a bunch of fields.
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
}

#[lang = "get_context"]
pub unsafe fn get_context<'a, 'b>(cx: ResumeTy) -> &'a mut Context<'b> {
    // NOTE: the actual implementation looks different.
    mem::transmute(cx.0)
}

#[lang = "pin"]
pub struct Pin<T>(T);
