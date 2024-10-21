//@ check-pass
// issue: rust-lang/rust#102933

use std::future::Future;

pub trait Service {
    type Response;
    type Future: Future<Output = Self::Response>;
}

pub trait A1: Service<Response = i32> {}

pub trait A2: Service<Future = Box<dyn Future<Output = i32>>> + A1 {
    fn foo(&self) {}
}

pub trait B1: Service<Future = Box<dyn Future<Output = i32>>> {}

pub trait B2: Service<Response = i32> + B1 {
    fn foo(&self) {}
}

fn main() {
    let x: &dyn A2 = todo!();
    let x: &dyn B2 = todo!();
}
