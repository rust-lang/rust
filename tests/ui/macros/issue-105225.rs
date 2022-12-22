fn main() {
    let x = 10;
    println!("{x}", x);
    //~^ ERROR argument never used
    let y = 20;
    println!("{x} {y}", x, y);
    //~^ ERROR multiple unused formatting arguments
}
