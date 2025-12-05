#![feature(coroutines)]

fn foo(_b: &bool, _a: ()) {}

fn main() {
    #[coroutine]
    || {
        let b = true;
        foo(&b, yield); //~ ERROR
    };
}
