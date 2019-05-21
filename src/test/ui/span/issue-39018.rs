pub fn main() {
    let x = "Hello " + "World!";
    //~^ ERROR cannot be applied to type

    // Make sure that the span outputs a warning
    // for not having an implementation for std::ops::Add
    // that won't output for the above string concatenation
    let y = World::Hello + World::Goodbye;
    //~^ ERROR cannot be applied to type

    let x = "Hello " + "World!".to_owned();
    //~^ ERROR cannot be applied to type
}

enum World {
    Hello,
    Goodbye,
}

fn foo() {
    let a = String::new();
    let b = String::new();
    let c = "";
    let d = "";
    let e = &a;
    let _ = &a + &b; //~ ERROR binary operation
    let _ = &a + b; //~ ERROR binary operation
    let _ = a + &b; // ok
    let _ = a + b; //~ ERROR mismatched types
    let _ = e + b; //~ ERROR binary operation
    let _ = e + &b; //~ ERROR binary operation
    let _ = e + d; //~ ERROR binary operation
    let _ = e + &d; //~ ERROR binary operation
    let _ = &c + &d; //~ ERROR binary operation
    let _ = &c + d; //~ ERROR binary operation
    let _ = c + &d; //~ ERROR binary operation
    let _ = c + d; //~ ERROR binary operation
}
