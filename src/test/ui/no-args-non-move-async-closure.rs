// edition:2018

#![feature(arbitrary_self_types, async_await, await_macro, futures_api, pin)]

fn main() {
    let _ = async |x: u8| {};
    //~^ ERROR `async` non-`move` closures with arguments are not currently supported
}
