use std::cell::RefCell;

fn foo(cell: &RefCell<Option<Vec<()>>>) {
    cell.borrow_mut().unwrap().pop().unwrap();
    //~^ ERROR cannot move out of dereference of `RefMut<'_, Option<Vec<()>>>` [E0507]
    //~| NOTE move occurs because value has type `Option<Vec<()>>`, which does not implement the `Copy` trait
    //~| HELP consider borrowing the `Option`'s content
}

fn main() {
    let cell = RefCell::new(None);
    foo(&cell);
}