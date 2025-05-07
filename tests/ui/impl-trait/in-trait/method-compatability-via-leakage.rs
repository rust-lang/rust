//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

trait Trait {
    fn foo() -> impl Sized + Send;
}

impl Trait for u32 {
    fn foo() -> impl Sized {}
}

fn main() {}
