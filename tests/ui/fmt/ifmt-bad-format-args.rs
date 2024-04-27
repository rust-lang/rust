fn main() {
    format_args!(); //~ ERROR: requires at least a format string argument
    format_args!(|| {}); //~ ERROR: must be a string literal
}
