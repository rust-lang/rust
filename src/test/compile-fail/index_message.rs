fn main() {
    let z = ();
    log(debug, z[0]); //~ ERROR cannot index a value of type `()`
}
