//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ regex-error-pattern: attempt to calculate the power with overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

fn main() {
    let _x = 2i32.pow(1024);
}
