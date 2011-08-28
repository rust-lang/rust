// error-pattern: literal

fn main() {
    // #ifmt's first argument must be a literal.  Hopefully this
    // restriction can be eased eventually to just require a
    // compile-time constant.
    let x = #ifmt["a" + "b"];
}
