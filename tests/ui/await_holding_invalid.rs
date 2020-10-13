// edition:2018
#![warn(clippy::await_holding_lock, clippy::await_holding_refcell_ref)]

use std::cell::RefCell;
use std::sync::Mutex;

async fn bad_lock(x: &Mutex<u32>) -> u32 {
    let guard = x.lock().unwrap();
    baz().await
}

async fn good_lock(x: &Mutex<u32>) -> u32 {
    {
        let guard = x.lock().unwrap();
        let y = *guard + 1;
    }
    baz().await;
    let guard = x.lock().unwrap();
    47
}

async fn baz() -> u32 {
    42
}

async fn also_bad_lock(x: &Mutex<u32>) -> u32 {
    let first = baz().await;

    let guard = x.lock().unwrap();

    let second = baz().await;

    let third = baz().await;

    first + second + third
}

async fn not_good_lock(x: &Mutex<u32>) -> u32 {
    let first = baz().await;

    let second = {
        let guard = x.lock().unwrap();
        baz().await
    };

    let third = baz().await;

    first + second + third
}

#[allow(clippy::manual_async_fn)]
fn block_bad_lock(x: &Mutex<u32>) -> impl std::future::Future<Output = u32> + '_ {
    async move {
        let guard = x.lock().unwrap();
        baz().await
    }
}

async fn bad_refcell(x: &RefCell<u32>) -> u32 {
    let b = x.borrow();
    baz().await
}

async fn bad_mut_refcell(x: &RefCell<u32>) -> u32 {
    let b = x.borrow_mut();
    baz().await
}

async fn good_refcell(x: &RefCell<u32>) -> u32 {
    {
        let b = x.borrow_mut();
        let y = *b + 1;
    }
    baz().await;
    let b = x.borrow_mut();
    47
}

async fn also_bad_refcell(x: &RefCell<u32>) -> u32 {
    let first = baz().await;

    let b = x.borrow_mut();

    let second = baz().await;

    let third = baz().await;

    first + second + third
}

async fn less_bad_refcell(x: &RefCell<u32>) -> u32 {
    let first = baz().await;

    let b = x.borrow_mut();

    let second = baz().await;

    drop(b);

    let third = baz().await;

    first + second + third
}

async fn not_good_refcell(x: &RefCell<u32>) -> u32 {
    let first = baz().await;

    let second = {
        let b = x.borrow_mut();
        baz().await
    };

    let third = baz().await;

    first + second + third
}

#[allow(clippy::manual_async_fn)]
fn block_bad_refcell(x: &RefCell<u32>) -> impl std::future::Future<Output = u32> + '_ {
    async move {
        let b = x.borrow_mut();
        baz().await
    }
}

fn main() {
    {
        let m = Mutex::new(100);
        good_lock(&m);
        bad_lock(&m);
        also_bad_lock(&m);
        not_good_lock(&m);
        block_bad_lock(&m);
    }
    {
        let rc = RefCell::new(100);
        good_refcell(&rc);
        bad_refcell(&rc);
        bad_mut_refcell(&rc);
        also_bad_refcell(&rc);
        less_bad_refcell(&rc);
        not_good_refcell(&rc);
        block_bad_refcell(&rc);
    }
}
