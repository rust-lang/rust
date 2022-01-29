// run-pass
use std::rc::Rc;

struct Foo<'r>(&'r mut i32);

impl<'r> Drop for Foo<'r> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut drops = 0;

    {
        let _: Rc<dyn Send> = Rc::new(Foo(&mut drops));
    }

    assert_eq!(1, drops);
}
