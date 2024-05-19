//@ edition:2021

#![feature(async_closure)]

fn main() {
    let _ = ||{}();
    //~^ ERROR expected function, found `()`

    let _ = async ||{}();
    //~^ ERROR expected function, found `()`
}
