// Check that we *don't* strip shebang in files that were `include`d in an expression or
// expression statement context.
// We do that to be consistent with frontmatter (see test `frontmatter/include-in-expr-ctxt.rs`).
// While there could be niche use cases for such shebang, it seems more confusing than beneficial.

fn main() {
    // expr ctxt
    _ = include!("auxiliary/shebang-expr.rs");
    //~^ ERROR non-expression macro in expression position
    //~? ERROR expected `[`, found `/`

    // stmt ctxt (reuses expr expander)
    include!("auxiliary/shebang-expr.rs");
    //~^ ERROR non-statement macro in statement position
    //~? ERROR expected `[`, found `/`
}
