//@ run-rustfix

fn main() {
    demo = 1; //~ ERROR cannot find value `demo`
    dbg!(demo); //~ ERROR cannot find value `demo`

    x = "x"; //~ ERROR cannot find value `x`
    println!("x: {}", x); //~ ERROR cannot find value `x`

    let_some_variable = 6; //~ cannot find value `let_some_variable`
    println!("some_variable: {}", some_variable); //~ ERROR cannot find value `some_variable`

    letother_variable = 6; //~ cannot find value `letother_variable`
    println!("other_variable: {}", other_variable); //~ ERROR cannot find value `other_variable`

    if x == "x" {
        //~^ ERROR cannot find value `x`
        println!("x is 1");
    }

    y = 1 + 2; //~ ERROR cannot find value `y`
    println!("y: {}", y); //~ ERROR cannot find value `y`
}
