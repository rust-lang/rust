//@ run-rustfix
#![allow(unused)]

struct S {
    f: String,
}

fn main() {
    let _moved @ _from = String::from("foo"); //~ ERROR
    let _moved @ ref _from = String::from("foo"); //~ ERROR
    let ref _moved @ _from = String::from("foo"); //~ ERROR
    //~^ ERROR
    let ref _moved @ ref _from = String::from("foo"); // ok
    let _moved @ S { f } = S { f: String::from("foo") }; //~ ERROR
    let ref _moved @ S { f } = S { f: String::from("foo") }; //~ ERROR
    //~^ ERROR
    let ref _moved @ S { ref f } = S { f: String::from("foo") }; // ok
    let _moved @ S { ref f } = S { f: String::from("foo") }; //~ ERROR
}
