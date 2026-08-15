// Ensure that there are straightforward ways to define "empty" / "trivial" / "unit" trait aliases
// which don't impose any constraints when used as a bound (since they expand to nothing).
//@ check-pass
#![feature(trait_alias)]

trait Empty =;

trait Trivial = where;

trait Unit = where Self:;

fn check<T: ?Sized + Empty>() {}

fn main() {
    check::<()>(); // OK. "`(): Empty`" is trivially satisfied
    check::<str>(); // OK. `Empty` is truly empty and isn't implicitly bounded by `Sized`.
}
