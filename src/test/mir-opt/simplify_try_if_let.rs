// compile-flags: -Zmir-opt-level=1
// EMIT_MIR rustc.{{impl}}-append.SimplifyArmIdentity.diff

use std::ptr::NonNull;

pub struct LinkedList {
    head: Option<NonNull<Node>>,
    tail: Option<NonNull<Node>>,
}

pub struct Node {
    next: Option<NonNull<Node>>,
}

impl LinkedList {
    pub fn new() -> Self {
        Self { head: None, tail: None }
    }

    pub fn append(&mut self, other: &mut Self) {
        match self.tail {
            None => { },
            Some(mut tail) => {
                // `as_mut` is okay here because we have exclusive access to the entirety
                // of both lists.
                if let Some(other_head) = other.head.take() {
                    unsafe {
                        tail.as_mut().next = Some(other_head);
                    }
                }
            }
        }
    }
}

fn main() {
    let mut one = LinkedList::new();
    let mut two = LinkedList::new();
    one.append(&mut two);
}
