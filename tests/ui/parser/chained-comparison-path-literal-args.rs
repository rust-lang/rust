// A path followed by `<literal>` and no further operand is not a chained comparison, so
// point at the turbofish and call syntaxes instead of only reporting the chaining.

fn main() {
    let _ = Some<1>;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments
    //~| HELP or use `(...)` if you meant to specify fn arguments
}

fn in_call_args() {
    drop(Some<1>);
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments
    //~| HELP or use `(...)` if you meant to specify fn arguments
}

fn genuine_comparison(x: u32) {
    // An operand follows the `>`, so no turbofish help here.
    let _ = x < 1 > 3;
    //~^ ERROR comparison operators cannot be chained
}
