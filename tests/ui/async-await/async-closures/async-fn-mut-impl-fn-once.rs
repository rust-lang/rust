//@ edition:2021
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn call_once<F>(_: impl FnOnce() -> F) {}

fn main() {
    let mut i = 0;
    let c = async || {
        i += 1;
    };
    call_once(c);
}
