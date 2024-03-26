//@ check-pass

#![feature(postfix_match)]

fn main() {
    (&1).match { a => a };
    (1 + 2).match { b => b };
}
