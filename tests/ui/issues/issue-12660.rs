//@ run-pass
//@ aux-build:issue-12660-aux.rs


extern crate issue12660aux;

use issue12660aux::{my_fn, MyStruct};

#[allow(path_statements)]
fn main() {
    my_fn(MyStruct);
    MyStruct;
}
