// xfail-stage0
// error-pattern:+ cannot be applied to type `{x: bool}`

fn main() { let x = {x: true} + {x: false}; }