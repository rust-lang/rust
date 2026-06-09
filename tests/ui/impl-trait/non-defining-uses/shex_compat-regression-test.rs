//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for trait-system-refactor-initiative#181.

struct ShExCompactPrinter;

struct TripleExpr;

impl ShExCompactPrinter {
    fn pp_triple_expr(&self) -> impl Fn(&TripleExpr, &ShExCompactPrinter) + '_ {
        move |te, printer| {
            printer.pp_triple_expr()(te, printer);
        }
    }
}
fn main() {}
