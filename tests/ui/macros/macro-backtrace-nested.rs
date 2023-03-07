// In expression position, but not statement position, when we expand a macro,
// we replace the span of the expanded expression with that of the call site.

macro_rules! nested_expr {
    () => (fake) //~ ERROR cannot find
    //~^ ERROR cannot find
}

macro_rules! call_nested_expr {
    () => (nested_expr!())
}

macro_rules! call_nested_expr_sum {
    () => { 1 + nested_expr!(); }
}

fn main() {
    1 + call_nested_expr!();
    call_nested_expr_sum!();
}
