// Ensure that we eagerly *expand* free alias types during variance computation.
//
// Since free alias types are always normalizable it's not unreasonable to expect that variance
// information "propagates through" free aliases unlike projections for example which constrain
// all of their generic arguments to be invariant[^1].
//
// For context, we can't *normalize* types before trying to compute variances because we need
// variances for normalization in the first place, more specifically type relating.
//
// [^1]: Parent args: Traits are invariant over their params. Own args: Projections can be rigid.
//
// issue: <https://github.com/rust-lang/rust/issues/114221>
//
//@ check-pass

// FIXME(lazy_type_alias): Revisit this before stabilization (it's not blocking tho):
//                         We might want to compute variances for free alias types again
//                         with a special rule. See `variance-1.rs` for details.

#![feature(lazy_type_alias)]

// `Co` is covariant over `'a` since we expand `A` to `&'a ()`.
struct Co<'a>(A<'a>);

// `A` is *not* in a variance relation with its args since it's a type alias.
type A<'a> = &'a ();

fn co<'a>(x: Co<'static>) {
    let _: Co<'a> = x; // OK
}

// `Contra` is contravariant over `'a` since we expand `B` to `fn(&'a ())`.
struct Contra<'a>(B<'a>);

// (not in a variance relation)
type B<'a> = fn(&'a ());

fn contra<'a>(x: Contra<'a>) {
    let _: Contra<'static> = x; // OK
}

// `CoContra` is covariant over `T` and contravariant over `U` since we expand `C`.
struct CoContra<T, U>(C<T, U>);

// (not in a variance relation)
type C<T, U> = Option<(T, fn(U))>;

fn co_contra<'a>(x: CoContra<&'static (), &'a ()>) -> CoContra<&'a (), &'static ()> {
    x // OK
}

// Check that we deeply expand:
struct Co2<'a>(D0<'a>);
type D0<'a> = D1<'a>;
type D1<'a> = D2<'a>;
type D2<'a> = D3<'a>;
type D3<'a> = &'a ();

fn co2<'a>(x: Co2<'static>) -> Co2<'a> { x } // OK

fn main() {}
