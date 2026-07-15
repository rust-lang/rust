//@ check-pass

#![deny(unused_braces)]

struct Error;
struct NotCopy;

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
}
