// xfail-stage0
// error-pattern:+ cannot be applied to type `rec(bool x)`

fn main() { let x = {x: true}; x += {x: false}; }