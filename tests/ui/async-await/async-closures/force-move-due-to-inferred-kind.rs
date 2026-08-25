//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

extern crate block_on;

fn force_fnonce<T: AsyncFnOnce()>(t: T) -> T { t }

fn main() {
    block_on::block_on(async {
        let x = 1i32;
        // `force_fnonce` pulls the closure's kind down to `FnOnce`,
        // which means that we don't treat the borrow of `x` as a
        // self-borrow (with `'env` lifetime). This leads to a lifetime
        // error which is solved by forcing the inner coroutine to
        // be `move` as well, so that it moves `x`.
        let c = force_fnonce(async move || {
            println!("{x}");
        });
        c().await;
    });
}
