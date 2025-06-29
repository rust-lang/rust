use std::rc::Rc;

fn bar<T: Send>(_: T) {}

fn main() {
    let x = Rc::new(5);
    bar(x);
    //~^ ERROR `Rc<{integer}>` cannot be sent between threads safely
}
