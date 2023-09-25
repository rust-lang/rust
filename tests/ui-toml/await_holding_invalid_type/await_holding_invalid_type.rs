#![warn(clippy::await_holding_invalid_type)]
use std::net::Ipv4Addr;

async fn bad() -> u32 {
    let _x = String::from("hello");
    baz().await
}

async fn bad_reason() -> u32 {
    let x = Ipv4Addr::new(127, 0, 0, 1);
    let y = baz().await;
    let _x = x;
    y
}

async fn good() -> u32 {
    {
        let _x = String::from("hi!");
        let _y = Ipv4Addr::new(127, 0, 0, 1);
    }
    baz().await;
    let _x = String::from("hi!");
    47
}

async fn baz() -> u32 {
    42
}

#[allow(clippy::manual_async_fn)]
fn block_bad() -> impl std::future::Future<Output = u32> {
    async move {
        let _x = String::from("hi!");
        baz().await
    }
}

fn main() {
    good();
    bad();
    bad_reason();
    block_bad();
}
