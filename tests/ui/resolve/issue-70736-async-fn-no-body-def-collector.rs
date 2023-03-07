// edition:2018

async fn free(); //~ ERROR without a body

struct A;
impl A {
    async fn inherent(); //~ ERROR without body
}

trait B {
    async fn associated();
    //~^ ERROR cannot be declared `async`
}
impl B for A {
    async fn associated(); //~ ERROR without body
    //~^ ERROR cannot be declared `async`
}

fn main() {}
