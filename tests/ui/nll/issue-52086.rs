use std::rc::Rc;
use std::sync::Arc;

struct Bar { field: Vec<i32> }

fn main() {
    let x = Rc::new(Bar { field: vec![] });
    drop(x.field);
//~^ ERROR cannot move out of an `Rc`

    let y = Arc::new(Bar { field: vec![] });
    drop(y.field);
//~^ ERROR cannot move out of an `Arc`
}
