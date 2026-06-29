// Demonstrate that free alias types don't constrain their lifetime & type arguments if the corresp.
// lifetime & type parameters are unused in the lazy type alias (before normalization) since we
// eagerly expand them during variance computation unlike other alias types which constrain args to
// be invariant.

// FIXME(lazy_type_alias): Revisit this before stabilization (altho it's not blocking):
//                         Do we want to compute variances for lazy type aliases & free alias types
//                         again and "force bivariant parameters to be invariant" if they're not
//                         constrained by a projection?
//                         This would make `struct WrapDiscard` below compile.

#![feature(lazy_type_alias)]

type Discard<'a, T> = ();

// `'a` and `T` are bivariant & unconstrained => rejection
struct WrapDiscard<'a, T>(Discard<'a, T>);
//~^ ERROR lifetime parameter `'a` is never used
//~| ERROR type parameter `T` is never used

type DiscardConstrained<'a, T, X> = X
where
    X: Iterator<Item = (&'a (), T)>;

// `'a` and `T` are bivariant & constrained => acceptance
struct WrapDiscardConstrained<'a, T, X>(DiscardConstrained<'a, T, X>)
where
    X: Iterator<Item = (&'a (), T)>;

type Co<'a> = std::vec::IntoIter<(&'a (), &'a ())>;

// NOTE: If we end up switching back to computing variances for free alias types with the special
//       rule explained in the FIXME above, then this function should still compile since
//       LTA `DiscardConstrained` should be bivariant over `'a` and `T`, not invariant due to them
//       being constrained by a projection.
fn bi<'r>(
    x: WrapDiscardConstrained<'static, &'static (), Co<'static>>,
) -> WrapDiscardConstrained<'r, &'r (), Co<'r>> {
    x
}

fn main() {}
