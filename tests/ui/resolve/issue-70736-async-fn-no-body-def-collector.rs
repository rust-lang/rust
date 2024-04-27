//@ edition:2018

async fn free(); //~ ERROR without a body

struct A;
impl A {
    async fn inherent(); //~ ERROR without body
}

trait B {
    async fn associated();
}
impl B for A {
    async fn associated(); //~ ERROR without body
}

fn main() {}
