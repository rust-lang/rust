fn main() {
    let x = 0;
    let y = 0;

    println!("{x}", x);
    //~^ ERROR: redundant argument

    println!("{x} {}", x, x);
    //~^ ERROR: redundant argument

    println!("{} {x}", x, x);
    //~^ ERROR: redundant argument

    println!("{x} {y}", x, y);
    //~^ ERROR: redundant arguments

    println!("{} {} {x} {y} {}", x, x, x, y, y);
    //~^ ERROR: redundant arguments
}
