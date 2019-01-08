fn foo() {
    (0..13).collect<Vec<i32>>();
    //~^ ERROR chained comparison
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `i32`
    //~| ERROR attempted to take value of method `collect`
}

fn bar() {
    Vec<i32>::new();
    //~^ ERROR chained comparison
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `i32`
    //~| ERROR cannot find function `new` in the crate root
}

fn qux() {
    (0..13).collect<Vec<i32>();
    //~^ ERROR chained comparison
    //~| ERROR chained comparison
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `i32`
    //~| ERROR attempted to take value of method `collect`
    //~| ERROR mismatched types
}

fn main() {}
