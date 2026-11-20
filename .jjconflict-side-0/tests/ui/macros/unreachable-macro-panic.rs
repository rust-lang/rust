//@ run-fail
//@ error-pattern:internal error: entered unreachable code
//@ needs-subprocess

fn main() {
    unreachable!()
}
