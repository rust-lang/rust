use std::rc::Rc;

fn foo(_x: Rc<usize>) {}

fn bar<F:FnOnce() + Send>(_: F) { }

fn main() {
    let x = Rc::new(3);
    bar(move|| foo(x));
    //~^ ERROR `std::rc::Rc<usize>` cannot be sent between threads safely
}
