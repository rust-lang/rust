fn match_not_diverging_block() {
    // Due to never-to-any coercion, the type of `{ return }` is a fresh type variable. Since we
    // can't resolve its type immediately, we can't negate it. `!` has a `Not` impl, so if the
    // pre-fallback type of `{ return }` was `!` instead, this would compile.
    match !{ return } {}
    //~^ ERROR type annotations needed
}

fn if_not_diverging_block() {
    // Here, `if` uses an expected type of `bool` for its condition, which we previously propagated
    // to the `!` operator's operand; see <https://github.com/rust-lang/rust/issues/151202>. To
    // prevent breakage in fixing that bug, we currently still accept this.
    // TODO: fcw
    if !{ return } {}
}

fn main() {}
