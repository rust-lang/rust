fn main() {
    let (foo @ ..,) = (0, 0);
    //~^ ERROR: `foo @` is not allowed in a tuple
    //~| ERROR: `..` patterns are not allowed here
    //~| ERROR: mismatched types
    dbg!(foo);
}
