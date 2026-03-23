//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ regex-error-pattern: attempt to compute.*\*.* which would overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

fn main() {
    let _x = 2i32.pow(1024);
}
