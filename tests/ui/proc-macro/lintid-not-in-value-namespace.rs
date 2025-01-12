//@ proc-macro: lint-id.rs

extern crate lint_id;

fn main() {
    eprintln!("{:?}", lint_id::ambiguous_thing);
    //~^ expected value, found warning id `lint_id::ambiguous_thing`
}
