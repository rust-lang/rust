//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    Some("foo").unwrap_or(panic!("bad input")).to_string();
}
