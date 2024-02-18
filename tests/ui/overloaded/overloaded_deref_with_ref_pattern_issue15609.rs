//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that we choose Deref or DerefMut appropriately based on mutability of ref bindings (#15609).

fn main() {
    use std::cell::RefCell;

    struct S {
        node: E,
    }

    enum E {
        Foo(u32),
        Bar,
    }

    // Check match
    let x = RefCell::new(S { node: E::Foo(0) });

    let mut b = x.borrow_mut();
    match b.node {
        E::Foo(ref mut n) => *n += 1,
        _ => (),
    }

    // Check let
    let x = RefCell::new(0);
    let mut y = x.borrow_mut();
    let ref mut z = *y;

    fn foo(a: &mut RefCell<Option<String>>) {
        if let Some(ref mut s) = *a.borrow_mut() {
            s.push('a')
        }
    }
}
