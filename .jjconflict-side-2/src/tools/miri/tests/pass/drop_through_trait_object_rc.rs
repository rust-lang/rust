trait Foo {}

struct Bar;

static mut DROP_CALLED: bool = false;

impl Drop for Bar {
    fn drop(&mut self) {
        unsafe {
            DROP_CALLED = true;
        }
    }
}

impl Foo for Bar {}

use std::rc::Rc;

fn main() {
    let b: Rc<dyn Foo> = Rc::new(Bar);
    assert!(unsafe { !DROP_CALLED });
    drop(b);
    assert!(unsafe { DROP_CALLED });
}
