//@ run-fail
//@ check-run-results
//@ needs-subprocess

const fn test(x: usize) -> i32 {
    [42;5][x]
}

fn main () {
    let _ = test(5);
}
