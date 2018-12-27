// aux-build:issue-30123-aux.rs

extern crate issue_30123_aux;
use issue_30123_aux::*;

fn main() {
    let ug = Graph::<i32, i32>::new_undirected();
    //~^ ERROR no function or associated item named `new_undirected` found for type
}
