//@error-in-other-file: memory leaked
//@normalize-stderr-test: ".*│.*" -> "$$stripped$$"
//@normalize-stderr-test: "Rust heap, size: [0-9]+, align: [0-9]+" -> "Rust heap, SIZE, ALIGN"

use std::cell::RefCell;
use std::rc::Rc;

struct Dummy(Rc<RefCell<Option<Dummy>>>);

fn main() {
    let x = Dummy(Rc::new(RefCell::new(None)));
    let y = Dummy(x.0.clone());
    *x.0.borrow_mut() = Some(y);
}
