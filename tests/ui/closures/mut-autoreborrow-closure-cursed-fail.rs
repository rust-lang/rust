//! This cursed example should compile,
//! but doesn't because we have to support `mut-autoreborrow-closure-cursed`.
//! Once we can get closures to implement `Reborrow`
//! and `FnMut` to be implemented for `Reborrow` closures,
//! we will be able to fix this.
//! See https://github.com/rust-lang/rust/issues/47478

fn assert_fnmut(_: &mut impl FnMut()) {}

fn main() {
    let x = &mut Box::new(0);
    let mut f = || {
        // Cursed HIR-inserted reborrow attempts to make this closure `FnMut`,
        // but at the cost of never letting us return `y`.
        let y: &mut _ = x;
        y //~ ERROR captured variable cannot escape `FnMut` closure body
    };
    f();
}
