// Check that bogus field access is non-fatal
fn main() {
    let x = 0;
    log(debug, x.foo); //! ERROR attempted access of field
    log(debug, x.bar); //! ERROR attempted access of field
}