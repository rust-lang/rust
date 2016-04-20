#![feature(plugin)]
#![plugin(clippy)]

use std::sync::Arc;

use std::mem::forget as forgetSomething;
use std::mem as memstuff;

#[deny(mem_forget)]
fn main() {
    let five: i32 = 5;
    forgetSomething(five);
    //~^ ERROR usage of std::mem::forget

    let six: Arc<i32> = Arc::new(6);
    memstuff::forget(six);
    //~^ ERROR usage of std::mem::forget

    std::mem::forget(7);
    //~^ ERROR usage of std::mem::forget
}
