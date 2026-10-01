// A test for the way we handle higher-ranked subtyping and subtyping requirements.
//
// - `let y = x` creates a `Subtype` obligation that is deferred for later.
// - `w = a` sets the type of `x` to `Option<for<'a> fn(&'a ())>` and generalized
//   `z` first to `Option<_>` and then to `Option<fn(&'0 ())>`.
//  - The various subtyping obligations are then processed.
//
// Whether the `?x <: ?y` obligation incorrectly constrains `?y` to
// `Option<for<'a> fn(&'a ())>` is order dependent and passed with the old
// solver while breaking with the new one.
//
// Found when considering fixes to #117151

//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[old] check-pass

fn main() {
    let mut x = None;
    let y = x;
    let z = Default::default();
    let mut w = (&mut x, z, z);
    //[next]~^ ERROR: mismatched types
    let a = (&mut None::<fn(&())>, y, None::<fn(&'static ())>);
    w = a;
}
