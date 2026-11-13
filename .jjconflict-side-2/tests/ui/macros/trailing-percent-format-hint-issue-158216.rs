//@ check-fail
//@ compile-flags: --crate-type=lib

pub fn f(x: f64) {
    println!("{x:>8.2}%", "foo");
    //~^ ERROR argument never used
}
