//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![recursion_limit = "8"]

// Regression test for the <https://github.com/bevyengine/bevy/issues/25511>.
//
// We previously didn't apply the future compat warning if overflow was hiding
// an error. We need to do so to avoid ambiguity in method selection.

struct W<T>(T);
struct Indir<T>(T);

trait Trait {}
impl<T> Trait for W<T>
where
    Indir<T>: Trait
{}
impl<T: Trait> Trait for Indir<T> {}

trait Overflow {
    fn foo(&self) {}
}
impl<T: Trait> Overflow for T {}

trait Accepted {
    fn foo(&self) {}
}
impl<T> Accepted for T {}

fn main() {
    W(W(W(()))).foo();
    W(W(W(W(W(()))))).foo();
    //[next]~^ WARN: overflow evaluating the requirement `W<W<W<W<W<()>>>>>: Overflow`
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
