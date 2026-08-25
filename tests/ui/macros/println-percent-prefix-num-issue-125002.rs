fn main() {
    println!("%100000", 1);
    //~^ ERROR argument never used
    println!("%     65536", 1);
    //~^ ERROR argument never used
}
