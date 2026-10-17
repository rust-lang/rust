//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2024

// This test previously used the `is_copy_raw` query during
// HIR typeck, dropping the list of generators from the current
// body. This then caused a query cycle.

struct W<T>(*const T);

impl<T: Send> Clone for W<T> {
    fn clone(&self) -> Self { W(self.0) }
}

impl<T: Send> Copy for W<T> {}

fn main() {
    let coro = async {};
    let x = W(&raw const coro);
    let c = || {
        let x = x;
    };
}
