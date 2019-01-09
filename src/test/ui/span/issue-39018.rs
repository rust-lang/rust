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
