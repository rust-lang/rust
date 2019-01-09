fn main() {
    assert!();  //~ ERROR requires a boolean expression
    assert!(struct); //~ ERROR expected expression
    debug_assert!(); //~ ERROR requires a boolean expression
    debug_assert!(struct); //~ ERROR expected expression
}
