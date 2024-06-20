// Given an anon const `a`: `{ N }` and some anon const `b` which references the
// first anon const: `{ [1; a] }`. `b` should not have any generics as it is not
// a simple `N` argument nor is it a repeat expr count.
//
// On the other hand `b` *is* a repeat expr count and so it should inherit its
// parents generics as part of the `const_evaluatable_unchecked` fcw (#76200).
//
// In this specific case however `b`'s parent should be `a` and so it should wind
// up not having any generics after all. If `a` were to inherit its generics from
// the enclosing item then the reference to `a` from `b` would contain generic
// parameters not usable by `b` which would cause us to ICE.

fn bar<const N: usize>() {}

fn foo<const N: usize>() {
    bar::<{ [1; N] }>();
    //~^ ERROR: generic parameters may not be used in const operations
}

fn main() {}
