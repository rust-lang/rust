fn main() {
    let x = 0;
    println!("{x}", x);
    //~^ ERROR: redundant argument

    println!("{x} {}", x, x);
    //~^ ERROR: redundant argument

    println!("{} {x}", x, x);
    //~^ ERROR: redundant argument

    let y = 0;
    println!("{x} {y}", x, y);
    //~^ ERROR: redundant argument

    let y = 0;
    println!("{} {} {x} {y} {}", x, x, x, y, y);
    //~^ ERROR: redundant argument
}
