//@ run-fail
//@ error-pattern:index out of bounds: the len is 5 but the index is 5
//@ needs-subprocess

const fn test(x: usize) -> i32 {
    [42;5][x]
}

fn main () {
    let _ = test(5);
}
