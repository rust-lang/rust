fn foo(x: &i32, y: &i32) -> &i32 {
    //~^ ERROR missing lifetime
    if x > y { x } else { y }
    //~^ ERROR lifetime of reference outlives
    //~| ERROR lifetime of reference outlives
}

fn main() {}
