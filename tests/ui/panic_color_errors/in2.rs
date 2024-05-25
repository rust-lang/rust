#![feature(panic_color_errors)]

fn foo(x: u64) {
    if x == 0 {
        panic!("Oops sometging went wrong");
    } else {
        foo(x-1);
    }
}

fn main() {
    std::panic::highlight_errors(true);
    foo(100);
}