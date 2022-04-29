trait Bar {}
impl Bar for u8 {}
fn foo() -> impl Bar {
    5; //~^ ERROR the trait bound `(): Bar` is not satisfied
}

fn main() {}
