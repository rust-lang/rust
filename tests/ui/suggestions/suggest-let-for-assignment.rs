//@ run-rustfix

fn main() {
    demo = 1; //~ ERROR cannot find value `demo` in this scope
    dbg!(demo); //~ ERROR cannot find value `demo` in this scope

    x = "x"; //~ ERROR cannot find value `x` in this scope
    println!("x: {}", x); //~ ERROR cannot find value `x` in this scope

    let_some_variable = 6; //~ ERROR cannot find value `let_some_variable` in this scope
    println!("some_variable: {}", some_variable); //~ ERROR cannot find value `some_variable` in this scope

    letother_variable = 6; //~ ERROR cannot find value `letother_variable` in this scope
    println!("other_variable: {}", other_variable); //~ ERROR cannot find value `other_variable` in this scope

    if x == "x" {
        //~^ ERROR cannot find value `x` in this scope
        println!("x is 1");
    }

    y = 1 + 2; //~ ERROR cannot find value `y` in this scope
    println!("y: {}", y); //~ ERROR cannot find value `y` in this scope
}
