fn main() {
    let x = 10;
    println!("{x}", x);
    //~^ ERROR argument never used
    let y = 20;
    println!("{x} {y}", x, y);
    //~^ ERROR multiple unused formatting arguments
    println!("{x} {y}", y, x);
    //~^ ERROR multiple unused formatting arguments
    println!("{} {y}", x, y);
    //~^ ERROR argument never used
    println!("{} {} {y} {} {}", y, y, y, y, y);
    //~^ ERROR argument never used
    println!("{x} {x} {y} {x} {x}", x, y, x, y, y);
    //~^ ERROR argument never used
}
