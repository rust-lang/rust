pub fn main() {
    let x = () + (); //~ ERROR binary operation

    // this shouldn't have a flow-on error:
    for _ in x {}
}
