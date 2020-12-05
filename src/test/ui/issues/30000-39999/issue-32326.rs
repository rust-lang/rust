// Regression test for #32326. We ran out of memory because we
// attempted to expand this case up to the recursion limit, and 2^N is
// too big.

enum Expr { //~ ERROR E0072
    Plus(Expr, Expr),
    Literal(i64),
}

fn main() { }
