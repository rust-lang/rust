// run-rustfix
use std::pin::Pin;

fn foo(_: &mut ()) {}

fn main() {
    let mut uwu = ();
    let mut r = Pin::new(&mut uwu);
    foo(r.get_mut());
    foo(r.get_mut()); //~ ERROR use of moved value
}
