//@ build-pass
//@ edition: 2024
// test that a recursive type in an async block does
// not cause a cycle.

pub struct A(Box<B>);
pub struct B(A);
pub async fn run(_: A) {}

pub enum A2 {
    X(Box<B2>),
}

pub enum B2 {
    Y(A2),
}
pub async fn run2(a: Option<A2>) {
    async {}.await;
    drop(a);
}

fn main() {}
