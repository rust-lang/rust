//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/197>.
// This is only an issue in the new solver, but I'm testing it in both solvers for now.
// This has to do with the fact that the recursive `walk_dir` is a revealing use, which has not
// yet been constrained from the defining use by the time that closure signature inference is
// performed. We don't really care, though, since anywhere we structurally match on a type in
// upvar analysis, we already call `structurally_resolve_type` right before `.kind()`.

fn walk_dir(cb: &()) -> impl Sized {
    || {
        let fut = walk_dir(cb);
    };
}

fn main() {}
