//@ run-fail
//@ error-pattern:panic 1
//@ needs-subprocess

fn main() {
    let x = 2;
    let y = &x;
    panic!("panic 1");
}
