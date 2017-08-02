// FIXME: Due to https://github.com/rust-lang/rust/issues/43457 we have to disable validation
// compile-flags: -Zmir-emit-validate=0

//error-pattern: the evaluated program leaked memory

use std::rc::Rc;
use std::cell::RefCell;

struct Dummy(Rc<RefCell<Option<Dummy>>>);

fn main() {
    let x = Dummy(Rc::new(RefCell::new(None)));
    let y = Dummy(x.0.clone());
    *x.0.borrow_mut() = Some(y);
}
