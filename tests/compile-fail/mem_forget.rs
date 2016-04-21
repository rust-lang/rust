#![feature(plugin)]
#![plugin(clippy)]

use std::sync::Arc;
use std::rc::Rc;

use std::mem::forget as forgetSomething;
use std::mem as memstuff;

#[deny(mem_forget)]
fn main() {
    let five: i32 = 5;
    forgetSomething(five);

    let six: Arc<i32> = Arc::new(6);
    memstuff::forget(six);
    //~^ ERROR usage of mem::forget on Drop type

    let seven: Rc<i32> = Rc::new(7);
    std::mem::forget(seven);
    //~^ ERROR usage of mem::forget on Drop type

    let eight: Vec<i32> = vec![8];
    forgetSomething(eight);
    //~^ ERROR usage of mem::forget on Drop type

    std::mem::forget(7);
}
