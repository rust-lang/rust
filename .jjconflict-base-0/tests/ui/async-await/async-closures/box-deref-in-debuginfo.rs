//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass

extern crate block_on;

pub trait Trait {
    fn callback(&mut self);
}
impl Trait for (i32,) {
    fn callback(&mut self) {
        println!("hi {}", self.0);
        self.0 += 1;
    }
}

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

async fn run(mut loader: Box<dyn Trait>) {
    let f = async move || {
        loader.callback();
        loader.callback();
    };
    call_once(f).await;
}

fn main() {
    block_on::block_on(async {
        run(Box::new((42,))).await;
    });
}
