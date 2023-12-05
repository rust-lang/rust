// edition: 2024
// compile-flags: -Zunstable-options
// check-pass

#![feature(gen_blocks, async_iterator)]

async fn bar() {}

async gen fn foo() {
    yield bar().await;
}

fn main() {}
