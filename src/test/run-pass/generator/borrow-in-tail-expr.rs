// run-pass

#![feature(generators)]

fn main() {
    let _a = || {
        yield;
        let a = String::new();
        a.len()
    };
}
