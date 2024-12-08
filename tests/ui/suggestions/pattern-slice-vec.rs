// Regression test for #87017.

//@ run-rustfix

fn main() {
    fn foo() -> Vec<i32> { vec![1, 2, 3] }

    if let [_, _, _] = foo() {}
    //~^ ERROR: expected an array or slice
    //~| HELP: consider slicing here

    if let [] = &foo() {}
    //~^ ERROR: expected an array or slice
    //~| HELP: consider slicing here

    if let [] = foo() {}
    //~^ ERROR: expected an array or slice
    //~| HELP: consider slicing here

    let v = vec![];
    match &v {
    //~^ HELP: consider slicing here
        [5] => {}
        //~^ ERROR: expected an array or slice
        _ => {}
    }

    let [..] = vec![1, 2, 3];
    //~^ ERROR: expected an array or slice
    //~| HELP: consider slicing here
}
