fn main() {
    format!(); //~ ERROR requires at least a format string argument
    format!(struct); //~ ERROR expected expression
    format!("s", name =); //~ ERROR expected expression
    format!("s", foo = foo, bar); //~ ERROR expected `=`
    format!("s", foo = struct); //~ ERROR expected expression
    format!("s", struct); //~ ERROR expected expression

    // This error should come after parsing errors to ensure they are non-fatal.
    format!(123); //~ ERROR format argument must be a string literal
}
