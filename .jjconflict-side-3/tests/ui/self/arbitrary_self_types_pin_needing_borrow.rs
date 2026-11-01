use std::pin::Pin;
struct S;

impl S {
    fn x(self: Pin<&mut Self>) {
    }
}

fn main() {
    Pin::new(S).x();
    //~^ ERROR the trait bound `S: Deref` is not satisfied
    //~| ERROR no method named `x` found for struct `Pin<Ptr>` in the current scope
}
