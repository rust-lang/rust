macro_rules! m {
    ($e:expr) => { "expr" };
    (a < $l:lifetime >) => { "lifetime" };
}

fn test_macro() {
    // Ensure turbofish recovery is disabled during macro matching to avoid spurious diagnostics.
    // The second arm does NOT match; the expr-fragment error is a hard error.
    let _ = m!(a < 'x >);
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>`
    //~| HELP or use `(...)`
    //~| ERROR expected `while`, `for`, `loop` or `{` after a label
}

fn main() {}
