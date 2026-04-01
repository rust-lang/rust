//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
fn foo() -> impl FnOnce(u32) -> u32 {
    |x| x.leading_zeros()
}

fn main() {}
