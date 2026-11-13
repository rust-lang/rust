//@ edition:2024

use core::marker::PhantomPinned;
use core::pin::pin;

fn a() {
    struct NotCopy<T>(T);
    #[allow(unused_mut)]
    let mut pointee = NotCopy(PhantomPinned);
    pin!(pointee);
    let _moved = pointee;
    //~^ ERROR use of moved value
}

fn b() {
    struct NotCopy<T>(T);
    let mut pointee = NotCopy(PhantomPinned);
    pin!(*&mut pointee);
    //~^ ERROR cannot move
    let _moved = pointee;
}

fn main() {
    a();
    b();
}
