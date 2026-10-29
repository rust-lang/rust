//@ edition: 2024

// regression test for https://github.com/rust-lang/rust/issues/132199.
// in edition 2024 impl Trait captures every in-scope lifetime by default, so
// E0700 only fires once an explicit use<..> restricts the set.

struct T;

impl T {
    // dup: opaque already has use<'_>, needs a second anonymous lifetime.
    // before the fix, it suggested use<'_, '_>, which doesnt compile.
    fn dup(&self, t: &T) -> impl Sized + use<'_> { (self, t) }
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds

    // single_anon: use<> with one anonymous lifetime still gets use<'_>.
    fn single_anon(&self) -> impl Sized + use<> { self }
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds

    // named: a named lifetime still gets suggested by name.
    fn named<'a>(&'a self, t: &'a T) -> impl Sized + use<> { (self, t) }
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
