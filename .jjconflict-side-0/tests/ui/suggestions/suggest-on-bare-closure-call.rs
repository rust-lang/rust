//@ edition:2021

fn main() {
    let _ = ||{}();
    //~^ ERROR expected function, found `()`

    let _ = async ||{}();
    //~^ ERROR expected function, found `()`
}
