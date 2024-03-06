fn foo(x: &i32, y: &i32) -> &i32 { //~ ERROR missing lifetime
    if x > y { x } else { y }
    //~^ ERROR: lifetime may not live long enough
    //~| ERROR: lifetime may not live long enough
}

fn main() {}
