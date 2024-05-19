use std::rc::Rc;
use std::sync::Arc;

use std::mem as memstuff;
use std::mem::forget as forgetSomething;

#[warn(clippy::mem_forget)]
#[allow(forgetting_copy_types)]
fn main() {
    let five: i32 = 5;
    forgetSomething(five);

    let six: Arc<i32> = Arc::new(6);
    memstuff::forget(six);
    //~^ ERROR: usage of `mem::forget` on `Drop` type
    //~| NOTE: argument has type `std::sync::Arc<i32>`

    let seven: Rc<i32> = Rc::new(7);
    std::mem::forget(seven);
    //~^ ERROR: usage of `mem::forget` on `Drop` type
    //~| NOTE: argument has type `std::rc::Rc<i32>`

    let eight: Vec<i32> = vec![8];
    forgetSomething(eight);
    //~^ ERROR: usage of `mem::forget` on `Drop` type
    //~| NOTE: argument has type `std::vec::Vec<i32>`

    let string = String::new();
    std::mem::forget(string);
    //~^ ERROR: usage of `mem::forget` on type with `Drop` fields
    //~| NOTE: argument has type `std::string::String`

    std::mem::forget(7);
}
