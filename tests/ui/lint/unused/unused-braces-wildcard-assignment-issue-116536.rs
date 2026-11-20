//@ run-pass

#![deny(unused_braces)]

use std::cell::Cell;

struct Error;
struct NotCopy;

struct DropCounter<'a>(&'a Cell<usize>);

impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

struct Pair<'a> {
    first: DropCounter<'a>,
    _second: DropCounter<'a>,
}

fn wildcard_assignment_moves_value() {
    let e = NotCopy;
    _ = { e };
}

fn tuple_wildcard_assignment_moves_value() {
    let e = (NotCopy, Error);
    (_, Error) = { e };
}

fn main() {
    wildcard_assignment_moves_value();
    tuple_wildcard_assignment_moves_value();

    let drops = Cell::new(0);

    let e = (DropCounter(&drops), DropCounter(&drops));
    (..) = { e };
    assert_eq!(drops.get(), 2);

    let e = [DropCounter(&drops), DropCounter(&drops)];
    [..] = { e };
    assert_eq!(drops.get(), 4);

    let e = (Error, DropCounter(&drops));
    (Error, ..) = { e };
    assert_eq!(drops.get(), 5);

    let first;
    let e = [DropCounter(&drops), DropCounter(&drops)];
    [first, ..] = { e };
    assert_eq!(drops.get(), 6);
    drop(first);
    assert_eq!(drops.get(), 7);

    let e = (Error, [DropCounter(&drops), DropCounter(&drops)]);
    (Error, [..]) = { e };
    assert_eq!(drops.get(), 9);

    let e = Pair { first: DropCounter(&drops), _second: DropCounter(&drops) };
    Pair { .. } = { e };
    assert_eq!(drops.get(), 11);

    let first;
    let e = Pair { first: DropCounter(&drops), _second: DropCounter(&drops) };
    Pair { first, .. } = { e };
    assert_eq!(drops.get(), 12);
    drop(first);
    assert_eq!(drops.get(), 13);
}
