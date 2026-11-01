//@ run-rustfix
//@ edition:2018
// Most of items are taken from ./recover-const-async-fn-ptr.rs but this is able to apply rustfix.

pub type T0 = const fn(); //~ ERROR an `fn` pointer type cannot be `const`
pub type T1 = const extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `const`
pub type T2 = const unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `const`
pub type T3 = async fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type T4 = async extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type T5 = async unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type T6 = const async unsafe extern "C" fn();
//~^ ERROR an `fn` pointer type cannot be `const`
//~| ERROR an `fn` pointer type cannot be `async`

pub type FTT0 = for<'a> const fn(); //~ ERROR an `fn` pointer type cannot be `const`
pub type FTT1 = for<'a> const extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `const`
pub type FTT2 = for<'a> const unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `const`
pub type FTT3 = for<'a> async fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type FTT4 = for<'a> async extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type FTT5 = for<'a> async unsafe extern "C" fn();
//~^ ERROR an `fn` pointer type cannot be `async`
pub type FTT6 = for<'a> const async unsafe extern "C" fn();
//~^ ERROR an `fn` pointer type cannot be `const`
//~| ERROR an `fn` pointer type cannot be `async`

// Tests with qualifiers in the wrong order
pub type W1 = unsafe const fn();
//~^ ERROR an `fn` pointer type cannot be `const`
pub type W2 = unsafe async fn();
//~^ ERROR an `fn` pointer type cannot be `async`
pub type W3 = for<'a> unsafe const fn();
//~^ ERROR an `fn` pointer type cannot be `const`

fn main() {}
