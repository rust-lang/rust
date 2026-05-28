//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for trait-system-refactor-initiative#196.
fn iterator(b: bool) -> impl Iterator<Item = String> {
    if b {
        // We need to eagerly figure out the type of `i` here by using
        // the `<opaque as IntoIterator>::Item` obligation. This means
        // we not only have to consider item bounds, but also blanket impls.
        for i in iterator(false) {
            i.len();
        }
    }

    vec![].into_iter()
}
fn main() {}
