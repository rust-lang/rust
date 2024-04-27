// Regression test for issue #91328.

//@ run-rustfix

#![allow(dead_code)]

fn foo(r: Result<Vec<i32>, i32>) -> i32 {
    match r {
    //~^ HELP: consider using `as_deref` here
        Ok([a, b]) => a + b,
        //~^ ERROR: expected an array or slice
        //~| NOTE: pattern cannot match with input type
        _ => 42,
    }
}

fn bar(o: Option<Vec<i32>>) -> i32 {
    match o {
    //~^ HELP: consider using `as_deref` here
        Some([a, b]) => a + b,
        //~^ ERROR: expected an array or slice
        //~| NOTE: pattern cannot match with input type
        _ => 42,
    }
}

fn baz(v: Vec<i32>) -> i32 {
    match v {
    //~^ HELP: consider slicing here
        [a, b] => a + b,
        //~^ ERROR: expected an array or slice
        //~| NOTE: pattern cannot match with input type
        _ => 42,
    }
}

fn qux(a: &Option<Box<[i32; 2]>>) -> i32 {
    match a {
    //~^ HELP: consider using `as_deref` here
        Some([a, b]) => a + b,
        //~^ ERROR: expected an array or slice
        //~| NOTE: pattern cannot match with input type
        _ => 42,
    }
}

fn main() {}
