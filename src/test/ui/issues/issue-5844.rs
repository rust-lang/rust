//aux-build:issue_5844_aux.rs

extern crate issue_5844_aux;

fn main () {
    issue_5844_aux::rand(); //~ ERROR: requires unsafe
}
