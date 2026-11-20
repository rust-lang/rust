// Test for specific details of how we handle higher-ranked subtyping to make
// sure that any changes are made deliberately.
//
// - `let y = x` creates a `Subtype` obligation that is deferred for later.
// - `w = a` sets the type of `x` to `Option<for<'a> fn(&'a ())>` and generalizes
//   `z` first to `Option<_>` and then to `Option<fn(&'0 ())>`.
//  - The various subtyping obligations are then processed.
//
// This requires that
// 1. the `Subtype` obligation from `y = x` isn't processed while the types of
//    `w` and `a` are being unified.
// 2. the pending subtype obligation isn't considered when determining the type
//    to generalize `z` to first (when related to the type of `y`).
//
// Found when considering fixes to #117151
//@ check-pass

fn main() {
    let mut x = None;
    let y = x;
    let z = Default::default();
    let mut w = (&mut x, z, z);
    let a = (&mut None::<fn(&())>, y, None::<fn(&'static ())>);
    w = a;
}
