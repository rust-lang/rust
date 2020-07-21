// run-rustfix
// edition:2018
// Most of items are taken from ./recover-const-async-fn-ptr.rs but this is able to apply rustfix.

pub type T3 = async fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type T4 = async extern fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type T5 = async unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`

pub type FTT3 = for<'a> async fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type FTT4 = for<'a> async extern fn(); //~ ERROR an `fn` pointer type cannot be `async`
pub type FTT5 = for<'a> async unsafe extern "C" fn();
//~^ ERROR an `fn` pointer type cannot be `async`

fn main() {}
