fn main() {
    match (0, 1, 2) {
        (pat, ..,) => {}
        //~^ ERROR trailing comma is not permitted after `..`
    }
}
