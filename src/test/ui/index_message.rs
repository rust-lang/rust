fn main() {
    let z = ();
    let _ = z[0]; //~ ERROR cannot index into a value of type `()`
}
