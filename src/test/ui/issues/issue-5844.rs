//aux-build:issue-5844-aux.rs

extern crate issue_5844_aux;

fn main () {
    issue_5844_aux::rand(); //~ ERROR: requires unsafe
}
