fn main() {
    let mut a = "a";
    a += { "b" };
    //~^ ERROR: binary assignment operation `+=` cannot be applied
}
