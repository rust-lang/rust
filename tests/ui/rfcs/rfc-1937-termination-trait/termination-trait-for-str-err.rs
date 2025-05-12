//@ run-fail
//@ error-pattern: An error message for you
//@ failure-status: 1
//@ needs-subprocess

fn main() -> Result<(), &'static str> {
    Err("An error message for you")
}
