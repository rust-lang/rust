fn main() {
    let x = "x";
    let y = "y";

    println!("{x}", x, x = y);
    //~^ ERROR: redundant argument

    println!("{x}", x = y, x = y);
    //~^ ERROR: duplicate argument named `x`
}
