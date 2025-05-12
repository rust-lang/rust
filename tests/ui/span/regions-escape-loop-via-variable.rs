fn main() {
    let x = 3;

    // Here, the variable `p` gets inferred to a type with a lifetime
    // of the loop body.  The regionck then determines that this type
    // is invalid.
    let mut p = &x;

    loop {
        let x = 1 + *p;
        p = &x;
    }
    //~^^ ERROR `x` does not live long enough
}
