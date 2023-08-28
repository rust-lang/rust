fn main() {
    let x = 0;
    println!("{x}", x);
    //~^ ERROR: argument never used

    println!("{x} {}", x, x);
    //~^ ERROR: argument never used

    println!("{} {x}", x, x);
    //~^ ERROR: argument never used

    let y = 0;
    println!("{x} {y}", x, y);
    //~^ ERROR: multiple unused formatting arguments

    let y = 0;
    println!("{} {} {x} {y} {}", x, x, x, y, y);
    //~^ ERROR: multiple unused formatting arguments
}
