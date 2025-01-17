//@ aux-build:block-on.rs
//@ run-pass
//@ check-run-results
//@ revisions: e2021 e2018
//@[e2018] edition:2018
//@[e2021] edition:2021

extern crate block_on;

async fn call_once(f: impl AsyncFnOnce()) { f().await; }

pub async fn async_closure(x: &mut i32) {
    let c = async move || {
        *x += 1;
    };
    call_once(c).await;
}

fn main() {
    block_on::block_on(async {
        let mut x = 0;
        async_closure(&mut x).await;
        assert_eq!(x, 1);
    });
}
