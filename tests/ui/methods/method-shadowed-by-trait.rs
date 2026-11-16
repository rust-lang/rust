// Show the diagnostic for cases where a method is expected by the user but a different one is
// shaddowing it causing a field access to fail. Test for #93726.

use std::cell::RefCell;
use std::rc::Rc;
use std::borrow::BorrowMut;

type NodeAlias = Option<Rc<RefCell<Node>>>;

#[derive(Clone, Debug, PartialEq)]
struct Node {
    value: String,
    next: NodeAlias,
}

#[derive(Clone, Debug, PartialEq)]
struct TransactionLog {
    head: NodeAlias,
    tail: NodeAlias,
    length: i32,
}

impl Node {
    fn new(value: String) -> Rc<RefCell<Node>> {
        Rc::new(RefCell::new(Node { value, next: None }))
    }
}

impl TransactionLog {
    pub fn empty_list() -> TransactionLog {
        TransactionLog {
            head: None,
            tail: None,
            length: 0,
        }
    }

    pub fn append(&mut self, value: String) {
        let node = Node::new(value);
        match self.tail.take() {
            Some(old_tail) => old_tail.borrow_mut().next = Some(node.clone()),
            //~^ ERROR: no field `next` on type `&mut Rc<RefCell<Node>>`
            //~| NOTE: unknown field
            //~| NOTE: this method is `std::borrow::BorrowMut::borrow_mut`
            //~| NOTE: you might have meant to call `std::cell::RefCell::<T>::borrow_mut` instead
            //~| NOTE: this returns `&mut Rc<RefCell<Node>>`
            None => self.head = Some(node.clone()),
        }
        self.length += 1;
        self.tail = Some(node);
    }
}


struct S;
struct Other {
    field: (),
}
impl Other {
    fn method(&self) {}
}

impl S {
    fn foo(&self) -> Other { Other { field: () } }
    //~^ NOTE: you might have meant to call
    //~| NOTE: you might have meant to call
    //~| NOTE: you might have meant to call
    //~| NOTE: you might have meant to call
}
mod a {
    use crate::{S, Other};
    pub trait T {
        fn foo(&self) {}
        //~^ NOTE: this returns `()`
        //~| NOTE: this returns `()`
        //~| NOTE: this returns `()`
        //~| NOTE: this returns `()`
    }
    impl T for S {}
    impl T for Box<S> {}
    pub trait K<P> {
        fn foo(&self) -> Other { Other { field : () } }
        //~^ NOTE: you might have meant to call
        //~| NOTE: you might have meant to call
        //~| NOTE: you might have meant to call
        //~| NOTE: you might have meant to call
    }
    impl K<()> for S {}
    impl K<i32> for Box<S> {}
    pub trait V {
        fn foo(&self) -> Other;
        //~^ NOTE: you might have meant to call
        //~| NOTE: you might have meant to call
        //~| NOTE: you might have meant to call
        //~| NOTE: you might have meant to call
    }
    impl V for Box<S> {
        fn foo(&self) -> Other { Other { field : () } }
    }
    pub trait U {
        fn foo(&self) {} // not suggested because its return type doesn't have the field
    }
    impl U for Box<S> {}
}
use a::T;

fn main() {
    let x = Box::new(S);
    x.foo().field; //~ ERROR: no field `field` on type `()`
    //~^ NOTE: unknown field
    //~| NOTE: this method is `a::T::foo`
    let y = x.foo();
    //~^ NOTE: this method is `a::T::foo`
    //~| NOTE: this method is `a::T::foo`
    let _ = y.field; //~ ERROR: no field `field` on type `()`
    //~^ NOTE: unknown field
    x.foo().method(); //~ ERROR: no method named `method` found for unit type `()` in the current scope
    //~^ NOTE: method not found in `()`
    //~| NOTE: this method is `a::T::foo`
    let _ = y.method(); //~ ERROR: no method named `method` found for unit type `()` in the current scope
    //~^ NOTE: method not found in `()`
}
