fn main() {
    match 0 {
        (pat, ..,) => {}
        //~^ ERROR trailing comma is not permitted after `..`
        //~| ERROR mismatched types
    }
}
