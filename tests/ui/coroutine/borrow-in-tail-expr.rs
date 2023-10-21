// run-pass

#![feature(coroutines)]

fn main() {
    let _a = || {
        yield;
        let a = String::new();
        a.len()
    };
}
