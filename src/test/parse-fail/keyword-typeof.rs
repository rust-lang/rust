// compile-flags: -Z parse-only

fn main() {
    let typeof = (); //~ ERROR expected pattern, found reserved keyword `typeof`
}
