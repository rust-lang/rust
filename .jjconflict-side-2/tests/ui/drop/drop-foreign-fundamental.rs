use std::ops::Deref;
use std::pin::Pin;

struct Whatever<T>(T);

impl<T> Deref for Whatever<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

struct A;

impl Drop for Pin<Whatever<A>> {
    //~^ ERROR  the `Drop` trait may only be implemented for local structs, enums, and unions
    fn drop(&mut self) {}
}

fn main() {
    let x = Pin::new(Whatever(1.0f32));
}
