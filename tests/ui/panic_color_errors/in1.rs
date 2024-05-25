#![feature(panic_color_errors)]

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
    std::panic::highlight_errors(true);
    foo(100);
}