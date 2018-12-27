// compile-flags: -Z parse-only

fn main() {
    let override = (); //~ ERROR expected pattern, found reserved keyword `override`
}
