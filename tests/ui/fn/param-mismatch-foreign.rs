extern "C" {
    fn foo(x: i32, y: u32, z: i32);
    //~^ NOTE function defined here
    //~| NOTE
}

fn main() {
    foo(1i32, 2i32);
    //~^ ERROR this function takes 3 arguments but 2 arguments were supplied
    //~| NOTE argument #2 of type `u32` is missing
    //~| HELP provide the argument
}
