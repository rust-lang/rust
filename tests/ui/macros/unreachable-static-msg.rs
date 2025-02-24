//@ run-fail
//@ error-pattern:internal error: entered unreachable code: uhoh
//@ needs-subprocess

fn main() {
    unreachable!("uhoh")
}
