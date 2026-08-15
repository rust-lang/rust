//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

#![feature(async_fn_traits)]

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let s = String::from("hello, world");
        let c = async move || {
            println!("{s}");
        };
        c().await;
        c().await;

        fn is_static<T: 'static>(_: &T) {}
        is_static(&c);

        // Check that `<{async fn} as AsyncFnOnce>::CallOnceFuture` owns its captures.
        fn call_once<F: AsyncFnOnce()>(f: F) -> F::CallOnceFuture { f() }
        is_static(&call_once(c));
    });
}
