#![warn(clippy::await_holding_refcell_ref)]

use std::cell::RefCell;

async fn bad(x: &RefCell<u32>) -> u32 {
    let b = x.borrow();
    baz().await
}

async fn bad_mut(x: &RefCell<u32>) -> u32 {
    let b = x.borrow_mut();
    baz().await
}

async fn good(x: &RefCell<u32>) -> u32 {
    {
        let b = x.borrow_mut();
        let y = *b + 1;
    }
    baz().await;
    let b = x.borrow_mut();
    47
}

async fn baz() -> u32 {
    42
}

async fn also_bad(x: &RefCell<u32>) -> u32 {
    let first = baz().await;

    let b = x.borrow_mut();

    let second = baz().await;

    let third = baz().await;

    first + second + third
}

async fn less_bad(x: &RefCell<u32>) -> u32 {
    let first = baz().await;

    let b = x.borrow_mut();

    let second = baz().await;

    drop(b);

    let third = baz().await;

    first + second + third
}

async fn not_good(x: &RefCell<u32>) -> u32 {
    let first = baz().await;

    let second = {
        let b = x.borrow_mut();
        baz().await
    };

    let third = baz().await;

    first + second + third
}

#[allow(clippy::manual_async_fn)]
fn block_bad(x: &RefCell<u32>) -> impl std::future::Future<Output = u32> + '_ {
    async move {
        let b = x.borrow_mut();
        baz().await
    }
}

fn main() {
    let rc = RefCell::new(100);
    good(&rc);
    bad(&rc);
    bad_mut(&rc);
    also_bad(&rc);
    less_bad(&rc);
    not_good(&rc);
    block_bad(&rc);
}
