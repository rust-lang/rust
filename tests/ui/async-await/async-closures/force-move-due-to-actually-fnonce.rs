//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

extern crate block_on;

fn consume(_: String) {}

fn main() {
    block_on::block_on(async {
        let x = 1i32;
        let s = String::new();
        // `consume(s)` pulls the closure's kind down to `FnOnce`,
        // which means that we don't treat the borrow of `x` as a
        // self-borrow (with `'env` lifetime). This leads to a lifetime
        // error which is solved by forcing the inner coroutine to
        // be `move` as well, so that it moves `x`.
        let c = async move || {
            println!("{x}");
            // This makes the closure FnOnce...
            consume(s);
        };
        c().await;
    });
}
