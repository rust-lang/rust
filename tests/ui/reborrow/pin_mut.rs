use std::pin::Pin;

fn method(a: Pin<&mut ()>) {}

fn main() {
    let a = &mut ();
    let a = Pin::new(a);
    let _ = method(a);
    let _ = method(a); //~ERROR use of moved value: `a`
}
