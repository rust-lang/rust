// FIXME: Description.

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
