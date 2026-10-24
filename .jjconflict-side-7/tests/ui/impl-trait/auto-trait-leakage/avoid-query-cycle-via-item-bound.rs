//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// When proving auto trait bounds, make sure that we depend on auto trait
// leakage if we can also prove it via an item bound.
fn is_send<T: Send>(_: T) {}

fn direct() -> impl Send {
    is_send(check(false)); // leaks auto traits, depends on `check`
    1u16
}

trait Indir: Send {}
impl Indir for u32 {}
fn indir() -> impl Indir {
    is_send(check(false)); // leaks auto traits, depends on `check`
    1u32
}

fn check(b: bool) -> impl Sized {
    if b {
        // must not leak auto traits, as we otherwise get a query cycle.
        is_send(direct());
        is_send(indir());
    }
    1u64
}

fn main() {
    check(true);
}
