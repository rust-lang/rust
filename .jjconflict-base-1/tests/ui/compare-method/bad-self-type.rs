use std::future::Future;
use std::task::{Context, Poll};

fn main() {}

struct MyFuture {}

impl Future for MyFuture {
    type Output = ();
    fn poll(self, _: &mut Context<'_>) -> Poll<()> {
    //~^ ERROR method `poll` has an incompatible type for trait
        todo!()
    }
}

trait T {
    fn foo(self);
    fn bar(self) -> Option<()>;
}

impl T for MyFuture {
    fn foo(self: Box<Self>) {}
    //~^ ERROR method `foo` has an incompatible type for trait
    fn bar(self) {}
    //~^ ERROR method `bar` has an incompatible type for trait
}
