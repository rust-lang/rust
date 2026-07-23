//@ run-pass

#![feature(coroutines)]

fn main() {
    let _a = #[coroutine] || {
        yield;
        let a = String::new();
        a.len()
    };
}
