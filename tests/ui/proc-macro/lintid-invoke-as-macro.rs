//@ proc-macro: lint-id.rs

extern crate lint_id;

lint_id::ambiguous_thing!();
//~^ expected macro, found lint id `lint_id::ambiguous_thing`

fn main() {}
