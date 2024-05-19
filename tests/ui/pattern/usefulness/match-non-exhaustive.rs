fn main() {
    match 0 { 1 => () } //~ ERROR non-exhaustive patterns
    match 0 { 0 if false => () } //~ ERROR non-exhaustive patterns
    //-| NOTE match arms with guards don't count towards exhaustivity
}
