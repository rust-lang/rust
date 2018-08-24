// aux-build:issue_16725.rs

extern crate issue_16725 as foo;

fn main() {
    unsafe { foo::bar(); }
    //~^ ERROR: function `bar` is private
}
