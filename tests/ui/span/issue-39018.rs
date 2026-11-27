pub fn main() {
    let x = "Hello " + "World!";
    //~^ ERROR can't add

    // Make sure that the span outputs a warning
    // for not having an implementation for std::ops::Add
    // that won't output for the above string concatenation
    let y = World::Hello + World::Goodbye;
    //~^ ERROR can't add

    let x = "Hello " + "World!".to_owned();
    //~^ ERROR can't add
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
    let _ = &a + &b; //~ ERROR can't add
    let _ = &a + b; //~ ERROR can't add
    let _ = a + &b; // ok
    let _ = a + b; //~ ERROR mismatched types
    let _ = e + b; //~ ERROR can't add
    let _ = e + &b; //~ ERROR can't add
    let _ = e + d; //~ ERROR can't add
    let _ = e + &d; //~ ERROR can't add
    let _ = &c + &d; //~ ERROR can't add
    let _ = &c + d; //~ ERROR can't add
    let _ = c + &d; //~ ERROR can't add
    let _ = c + d; //~ ERROR can't add
}
