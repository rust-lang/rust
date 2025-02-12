//@ edition: 2021

fn foo() {}

fn main() {
    let _: Vec<(&str, fn())> = [("foo", foo)].into_iter().collect(); //~ ERROR
    let _: Vec<fn()> = [foo].into_iter().collect(); //~ ERROR
    let _: Vec<fn()> = Vec::from([foo]); //~ ERROR
}
