pub fn main() {
    let x = () + (); //~ ERROR cannot add `()` to `()`

    // this shouldn't have a flow-on error:
    for _ in x {}
}
