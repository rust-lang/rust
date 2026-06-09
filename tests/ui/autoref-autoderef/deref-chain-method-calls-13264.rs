//! Regression test for https://github.com/rust-lang/rust/issues/13264

//@ run-pass
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ops::Deref;

struct Root {
    jsref: JSRef
}

impl Deref for Root {
    type Target = JSRef;

    fn deref<'a>(&'a self) -> &'a JSRef {
        &self.jsref
    }
}

#[derive(Copy, Clone)]
struct JSRef {
    node: *const Node
}

impl Deref for JSRef {
    type Target = Node;

    fn deref<'a>(&'a self) -> &'a Node {
        self.get()
    }
}

trait INode {
    fn RemoveChild(&self);
}

impl INode for JSRef {
    fn RemoveChild(&self) {
        self.get().RemoveChild(0)
    }
}

impl JSRef {
    fn AddChild(&self) {
        self.get().AddChild(0);
    }

    fn get<'a>(&'a self) -> &'a Node {
        unsafe {
            &*self.node
        }
    }
}

struct Node;

impl Node {
    fn RemoveChild(&self, _a: usize) {
    }

    fn AddChild(&self, _a: usize) {
    }
}

fn main() {
    let n = Node;
    let jsref = JSRef { node: &n };
    let root = Root { jsref: jsref };

    root.AddChild();
    jsref.AddChild();

    root.RemoveChild();
    jsref.RemoveChild();
}
