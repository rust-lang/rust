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
impl S {
    fn foo(&self) -> Other { Other { field: () } }
}
mod a {
    use crate::{S, Other};
    pub trait T {
        fn foo(&self) {}
    }
    impl T for S {}
    impl T for Box<S> {}
    pub trait K {
        fn foo(&self) -> Other { Other { field : () } }
    }
    impl K for S {}
    impl K for Box<S> {}
    pub trait V {
        fn foo(&self) -> Other;
    }
    impl V for Box<S> {
        fn foo(&self) -> Other { Other { field : () } }
    }
    pub trait U {
        fn foo(&self) {}
    }
    impl U for Box<S> {}
}
use a::T;

fn main() {
    let x = Box::new(S);
    x.foo().field; //~ ERROR: no field `field` on type `()`
    //~^ NOTE: unknown field
}
