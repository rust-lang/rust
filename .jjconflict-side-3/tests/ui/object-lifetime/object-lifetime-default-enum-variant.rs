//@ check-pass

enum Enum<'a, T: 'a + AbideBy<'a> + ?Sized> {
    Variant,

    Carry(&'a T),
}

fn scope<'any>() {
    // We elaborate `dyn Trait` to `dyn Trait + 'any` due to bound `'a` on `T`.
    let _ = Enum::Variant::<'any, dyn Trait> {};

    // Similarly here.
    let _ = Enum::<'any, dyn Trait>::Variant {};
}

trait Trait {}

// We use this to test that a given trait object lifetime bound is
// *exactly equal* to a given lifetime (not longer, not shorter).
trait AbideBy<'a> {}
impl<'a> AbideBy<'a> for dyn Trait + 'a {}

fn main() {}
