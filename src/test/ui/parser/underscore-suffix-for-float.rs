// compile-flags: -Z parse-only

fn main() {
    let a = 42._; //~ ERROR expected identifier, found reserved identifier `_`
}
