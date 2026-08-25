// Ensure that we anonymize the output of a function for tail call signature compatibility.

//@ check-pass

#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

fn foo() -> for<'a> fn(&'a ()) {
    become bar();
}

fn bar() -> for<'b> fn(&'b ()) {
    todo!()
}

fn main() {}
