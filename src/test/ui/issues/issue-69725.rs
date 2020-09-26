// aux-build:issue-69725.rs

extern crate issue_69725;
use issue_69725::Struct;

fn crash<A>() {
    let _ = Struct::<A>::new().clone();
    //~^ ERROR: the method
}

fn main() {}
