// FIXME: This test should pass as the first two fields add implied bounds that
// `'a` is equal to `'b` while the last one should simply use that fact. With
// the current implementation this errors. We have to be careful as implied bounds
// are only sound if they're also correctly checked.

struct Inv<T>(*mut T); // `T` is invariant.
type A = for<'a, 'b> fn(Inv<&'a &'b ()>, Inv<&'b &'a ()>, Inv<&'a ()>);
type B = for<'a, 'b> fn(Inv<&'a &'b ()>, Inv<&'b &'a ()>, Inv<&'b ()>);

fn main() {
    let x: A = |_, _, _| ();
    let y: B = x; //~ ERROR mismatched types
    let _: A = y; //~ ERROR mismatched types
}
