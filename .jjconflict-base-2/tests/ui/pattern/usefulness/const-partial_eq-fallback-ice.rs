#![allow(warnings)]

struct MyType;

impl PartialEq<usize> for MyType {
    fn eq(&self, y: &usize) -> bool {
        true
    }
}

const CONSTANT: &&MyType = &&MyType;

fn main() {
    if let CONSTANT = &&MyType {
        //~^ ERROR constant of non-structural type `MyType` in a pattern
        println!("did match!");
    }
}
