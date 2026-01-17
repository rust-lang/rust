fn match_not_diverging_block() {
    // Unary negation operators require the type of their operand to be known after checking it.
    // Since `{ return }` performs never-to-any coercion to a fresh inference variable, its type at
    // that time is that inference variable, rather than `!`, meaning we can't negate it. `!` has a
    // `Not` impl, so if the pre-fallback type of `{ return }` was `!`, this would compile.
    match !{ return } {}
    //~^ ERROR type annotations needed
}

fn if_not_diverging_block() {
    // This was previously accepted. Since the logical negation here is expected to be of type
    // `bool`, its operand was as well due to #151202, which guided inference.
    if !{ return } {}
}

fn main() {}
