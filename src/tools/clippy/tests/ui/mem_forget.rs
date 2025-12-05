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
    //~^ mem_forget

    let seven: Rc<i32> = Rc::new(7);
    std::mem::forget(seven);
    //~^ mem_forget

    let eight: Vec<i32> = vec![8];
    forgetSomething(eight);
    //~^ mem_forget

    let string = String::new();
    std::mem::forget(string);
    //~^ mem_forget

    std::mem::forget(7);
}
