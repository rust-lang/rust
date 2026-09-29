//! This cursed example should never have compiled
//! (well, not without `Reborrow`),
//! but it does, so we are stuck supporting it forever.
//! See https://github.com/rust-lang/rust/issues/47478
//@ run-pass

fn assert_fnmut(_: &mut impl FnMut()) {}

fn main() {
    let x = &mut Box::new(0);
    let mut f = || {
        // Cursed HIR-inserted reborrow makes this closure `FnMut`,
        // but at the cost of never letting us return `_y`.
        let _y: &mut _ = x;
    };
    f();
    f();
    assert_fnmut(&mut f);
}
