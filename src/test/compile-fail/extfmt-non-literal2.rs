// error-pattern: literal

// xfail-test (issue #936)


fn main() {
    // #fmt's first argument must be a literal.  Hopefully this
    // restriction can be eased eventually to just require a
    // compile-time constant.
    let x = #fmt[20];
}
