#![feature(panic_color_errors)]

//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0

static mut I: [u64; 2] = [0; 2];

fn foo(x: u64) {
    if x == 0 {
        unsafe{
            let j = 12;
            I[j] = 0;
        }
    } else {
        foo(x-1);
    }
}

fn main() {
    std::panic::highlight_errors(None);
    foo(100);
}
