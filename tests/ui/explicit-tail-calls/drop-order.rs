//@ run-pass
//@ ignore-backends: gcc
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]
use std::cell::RefCell;

fn main() {
    let tail_counter = Default::default();
    tail_recursive(0, &tail_counter);
    assert_eq!(tail_counter.into_inner(), (0..128).collect::<Vec<u8>>());

    let simply_counter = Default::default();
    simply_recursive(0, &simply_counter);
    assert_eq!(simply_counter.into_inner(), (0..128).rev().collect::<Vec<u8>>());

    let scope_counter = Default::default();
    out_of_inner_scope(&scope_counter);
    assert_eq!(scope_counter.into_inner(), (0..8).collect::<Vec<u8>>());
}

fn tail_recursive(n: u8, order: &RefCell<Vec<u8>>) {
    if n >= 128 {
        return;
    }

    let _local = DropCounter(n, order);

    become tail_recursive(n + 1, order)
}

fn simply_recursive(n: u8, order: &RefCell<Vec<u8>>) {
    if n >= 128 {
        return;
    }

    let _local = DropCounter(n, order);

    return simply_recursive(n + 1, order);
}

fn out_of_inner_scope(order: &RefCell<Vec<u8>>) {
    fn inner(order: &RefCell<Vec<u8>>) {
        let _7 = DropCounter(7, order);
        let _6 = DropCounter(6, order);
    }

    let _5 = DropCounter(5, order);
    let _4 = DropCounter(4, order);

    if true {
        let _3 = DropCounter(3, order);
        let _2 = DropCounter(2, order);
        loop {
            let _1 = DropCounter(1, order);
            let _0 = DropCounter(0, order);

            become inner(order);
        }
    }
}

struct DropCounter<'a>(u8, &'a RefCell<Vec<u8>>);

impl Drop for DropCounter<'_> {
    #[track_caller]
    fn drop(&mut self) {
        self.1.borrow_mut().push(self.0);
    }
}
