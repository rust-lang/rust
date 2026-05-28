//@ aux-build: foreign.rs

extern crate foreign;

fn main() {
    let mut x = vec![];
    let h = foreign::hello(&x);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE immutable borrow occurs here
    x.push(0);
    //~^ ERROR cannot borrow `x` as mutable
    //~| NOTE mutable borrow occurs here
    println!("{h}");
    //~^ NOTE immutable borrow later used here
}
