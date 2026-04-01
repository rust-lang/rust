fn main() {
    println!(3 + 4);
    //~^ ERROR format argument must be a string literal
    println!(3, 4);
    //~^ ERROR format argument must be a string literal
}
