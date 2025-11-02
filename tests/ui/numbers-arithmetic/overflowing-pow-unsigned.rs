//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ error-pattern: attempt to multiply with overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

fn main() {
    let _x = 2u32.pow(1024);
}
