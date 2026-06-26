//@ aux-build:block-on.rs
//@ edition:2018
//@ run-pass
//@ check-run-results

#![allow(unused)]

extern crate block_on;

struct DropMe(i32);

impl Drop for DropMe {
    fn drop(&mut self) {
        println!("{} was dropped", self.0);
    }
}

async fn call_once(f: impl AsyncFnOnce()) {
    println!("before call");
    let fut = Box::pin(f());
    println!("after call");
    drop(fut);
    println!("future dropped");
}

fn main() {
    block_on::block_on(async {
        let d = DropMe(42);
        let async_closure = async move || {
            let d = &d;
            println!("called");
        };

        call_once(async_closure).await;
        println!("after");
    });
}
