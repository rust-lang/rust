//@ proc-macro: lint-id.rs

extern crate lint_id;

lint_id::ambiguous_thing!();
//~^ expected macro, found warning id `lint_id::ambiguous_thing`

fn main() {}
