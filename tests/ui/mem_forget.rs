use std::rc::Rc;
use std::sync::Arc;

use std::mem as memstuff;
use std::mem::forget as forgetSomething;

#[warn(clippy::mem_forget)]
#[allow(clippy::forget_copy)]
fn main() {
    let five: i32 = 5;
    forgetSomething(five);

    let six: Arc<i32> = Arc::new(6);
    memstuff::forget(six);

    let seven: Rc<i32> = Rc::new(7);
    std::mem::forget(seven);

    let eight: Vec<i32> = vec![8];
    forgetSomething(eight);

    std::mem::forget(7);
}
