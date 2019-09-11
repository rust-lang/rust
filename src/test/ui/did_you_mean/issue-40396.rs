fn foo() {
    (0..13).collect<Vec<i32>>();
    //~^ ERROR chained comparison
}

fn bar() {
    Vec<i32>::new();
    //~^ ERROR chained comparison
}

fn qux() {
    (0..13).collect<Vec<i32>();
    //~^ ERROR chained comparison
}

fn main() {}
