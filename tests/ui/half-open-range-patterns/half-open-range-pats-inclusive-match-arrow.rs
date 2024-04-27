fn main() {
    let x = 42;
    match x {
        //~^ ERROR: non-exhaustive patterns
        //~| NOTE: not covered
        //~| NOTE: matched value is of type
        0..=73 => {},
        74..=> {},
        //~^ ERROR unexpected `>` after inclusive range
        //~| NOTE this is parsed as an inclusive range `..=`
    }
}
