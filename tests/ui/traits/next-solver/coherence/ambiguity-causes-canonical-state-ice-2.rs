//@ compile-flags: -Znext-solver=coherence

// A regression test for #124791. Computing ambiguity causes
// for the overlap of the `ToString` impls caused an ICE.
#![crate_type = "lib"]
trait ToOwned {
    type Owned;
}
impl<T> ToOwned for T {
    type Owned = u8;
}
impl ToOwned for str {
    type Owned = i8;
}

trait Overlap {}
impl<T: ToOwned<Owned = i8> + ?Sized> Overlap for T {}
impl Overlap for str {}
//~^ ERROR conflicting implementations of trait `Overlap`
