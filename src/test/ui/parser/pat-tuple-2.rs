// compile-flags: -Z parse-only

fn main() {
    match 0 {
        (pat, ..,) => {} //~ ERROR trailing comma is not permitted after `..`
    }
}
