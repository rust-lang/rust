#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    let mut my_var = false;
    let callback = use || {
        my_var = true;
    };
    callback();
    //~^ ERROR cannot borrow `callback` as mutable, as it is not declared as mutable [E0596]
}
