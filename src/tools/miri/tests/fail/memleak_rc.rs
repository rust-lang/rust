//@error-pattern: the evaluated program leaked memory
//@stderr-per-bitwidth
//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"

use std::cell::RefCell;
use std::rc::Rc;

struct Dummy(Rc<RefCell<Option<Dummy>>>);

fn main() {
    let x = Dummy(Rc::new(RefCell::new(None)));
    let y = Dummy(x.0.clone());
    *x.0.borrow_mut() = Some(y);
}
