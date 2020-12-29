// Needs build-pass to trigger `optimized_mir` on all mir bodies
// build-pass
// compile-flags: -Zmir-opt-level=2

#[inline(always)]
fn f(g: impl Fn()) {
    g();
}

#[inline(always)]
fn g() {
    f(main);
}

fn main() {
    f(g);
}
