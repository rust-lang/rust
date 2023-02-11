fn main() {
    let x = 42;
    match x {
        0..=73 => {},
        74..=> {},
        //~^ ERROR unexpected `>` after inclusive range
        //~| NOTE this is parsed as an inclusive range `..=`
    }
}
