// This test should never pass!

use std::cell::RefCell;
use std::rc::Rc;

trait Swap: Sized {
    fn swap(self, other: Self);
}

impl<T> Swap for Rc<RefCell<T>> {
    fn swap(self, other: Self) {
        <RefCell<T>>::swap(&self, &other);
    }
}

fn hide<'a, 'b: 'a, T: 'static>(x: Rc<RefCell<&'b T>>) -> impl Swap + 'a {
    x
    //~^ ERROR hidden type for `impl Swap + 'a` captures lifetime that does not appear in bounds
}

fn dangle() -> &'static [i32; 3] {
    let long = Rc::new(RefCell::new(&[4, 5, 6]));
    let x = [1, 2, 3];
    let short = Rc::new(RefCell::new(&x));
    hide(long.clone()).swap(hide(short));
    let res: &'static [i32; 3] = *long.borrow();
    res
}

fn main() {
    println!("{:?}", dangle());
}
