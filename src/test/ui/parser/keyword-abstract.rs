// compile-flags: -Z parse-only

fn main() {
    let abstract = (); //~ ERROR expected pattern, found reserved keyword `abstract`
}
