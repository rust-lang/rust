// error-pattern:cannot index a value of type `()`
fn main() {
    let z = ();
    log(error, z[0]);
}
