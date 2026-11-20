//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// A regression test making sure that where-bounds with concrete aliases
// which normalize to something mentioning a generic parameters are
// considered non-global.
//
// When checking this, we previously didn't recur into types if they didn't
// mention any generic parameters, causing us to consider the `(<() as Id>::This,): Id`
// where-bound as global, even though it normalizes to `(T,): Id`.

trait Id {
    type This;
}

impl<T> Id for T {
    type This = T;
}

fn foo<T>(x: <(*const T,) as Id>::This) -> (*const T,)
where
    (): Id<This = *const T>,
    (<() as Id>::This,): Id,
{
    x //~ ERROR mismatched types
}

fn main() {}
