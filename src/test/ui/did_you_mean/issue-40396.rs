fn foo() {
    println!("{:?}", (0..13).collect<Vec<i32>>()); //~ ERROR chained comparison
}

fn bar() {
    println!("{:?}", Vec<i32>::new()); //~ ERROR chained comparison
}

fn qux() {
    println!("{:?}", (0..13).collect<Vec<i32>()); //~ ERROR chained comparison
    //~^ ERROR chained comparison
}

fn main() {}
