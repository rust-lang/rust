//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/rust/issues/149379.

fn foo<T>(x: (T, ())) -> Box<T> {
    Box::new(x.0)
}

fn main() {
    // Uses expectation as its struct tail is sized, resulting in `(dyn Send, ())`
    let _: Box<dyn Send> = foo(((), ()));
}
