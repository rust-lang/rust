//@ add-minicore
//@ compile-flags: --target armv5te-none-eabi
//@ needs-llvm-components: arm
//@ ignore-backends: gcc
//@ edition: 2024

#![crate_type = "lib"]
#![feature(no_core, lang_items,)]
#![no_core]

extern crate minicore;
use minicore::*;




#[instruction_set(arm::a32)]
fn foo() {
}

#[instruction_set(arm)]
//~^ ERROR malformed `instruction_set` attribute input [E0539]
fn bar() {
}

#[instruction_set(arm::)]
//~^ ERROR expected identifier, found `<eof>`
fn bazz() {
}

#[instruction_set(arm::magic)]
//~^ ERROR malformed `instruction_set` attribute input [E0539]
fn bazzer() {

}

fn all_instruction_set_cases() {
    #[instruction_set(arm::a32)]
    || {
        0
    };
    #[instruction_set(arm::t32)]
    async || {
        0
    };
}

struct Fooer;

impl Fooer {
    #[instruction_set(arm::a32)]
    fn fooest() {

    }
}

trait Bazzest {
    fn bazz();

    #[instruction_set(arm::a32)]
    fn bazziest() {

    }
}
impl Bazzest for Fooer {
    #[instruction_set(arm::t32)]
    fn bazz() {}
}


// The following lang items need to be defined for the async closure to work
#[lang = "ResumeTy"]
pub struct ResumeTy(NonNull<Context<'static>>);

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
    unsafe {mem::transmute(cx.0)}
}

#[lang = "pin"]
pub struct Pin<T>(T);

#[lang = "future_from_coroutine"]
pub const fn future_from_coroutine<C: Coroutine<ResumeTy, Yield = ()>>(
    coroutine: C,
) -> CoroutineFuture<C> {
    CoroutineFuture(coroutine)
}

#[lang = "CoroutineFuture"]
pub struct CoroutineFuture<C>(C);

impl<C> Future for CoroutineFuture<C>
where
    C: Coroutine<ResumeTy, Yield = ()>,
{
    type Output = C::Return;

    // NOTE: misses the `poll` method.
}

#[lang = "coroutine_state"]
pub enum CoroutineState<Y, R> {
    Yielded(Y),
    Complete(R),
}

#[lang = "coroutine"]
pub trait Coroutine<R = ()> {
    #[lang = "coroutine_yield"]
    type Yield;

    #[lang = "coroutine_return"]
    type Return;

    // NOTE: misses the `resume` method.
}
