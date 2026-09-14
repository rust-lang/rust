//@ known-bug: #158797
#![feature(coroutines)]
#![feature(const_async_blocks)]
#![feature(yield_expr)]
enum Foo {
    Bar = (
        #[coroutine]
        || yield,
        2,
    )
        .1,
}

fn main() {}
