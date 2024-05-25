#![feature(panic_color_errors)]

//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0

fn foo(x: u64) {
    if x == 0 {
        panic!("Oops sometging went wrong");
    } else {
        foo(x-1);
    }
}

fn main() {
    std::panic::highlight_errors(Some(false));
    foo(100);
}
