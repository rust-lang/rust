//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// When proving `C::Val<'a>: for<'b> PartialEq<C::Val<'b>>` there are
// two candidates:
// - `C::Val<'a>: PartialEq<C::Val<'a>>`
// - `for<'b> C::Val<'a>: PartialEq<C::Val<'b>>`
//
// The first candidate results in an `'!b == 'a` constraint while the
// second one has no region constraints. The constraint of the first
// candidate means the canonical response has an entry for `'b` in its
// `var_values`. This previously caused us to not prefer the second
// constraint.
//
// FIXME(trait-system-refactor-initiative#305): Now, actually, the first
// candidate should just result in a leak check failure, but doesn't.
// That's a separate issue though :>
trait Cursor {
    type Val<'a>: PartialEq + for<'b> PartialEq<Self::Val<'b>>;
}
impl<C: Cursor> Cursor for &C {
    // for<'b> C::Val<'a>: PartialEq<C::Val<'b>>
    // -  C::Val<'a>: PartialEq<C::Val<'a>>`
    // -  for<'b> C::Val<'a>: PartialEq<C::Val<'b>>
    type Val<'a> = C::Val<'a>;
}
fn main() {}
