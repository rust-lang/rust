// check-pass
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

macro_rules! tests {
    () => {
        || 42
    }
}
fn main() {
    // these two expressions expand to the same HIR and THIR nodes,
    // yet the macro version has caused double-steal errors
    || 42;
    tests!();
}
