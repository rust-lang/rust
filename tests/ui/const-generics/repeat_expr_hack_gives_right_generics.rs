// Given a const argument `a`: `{ N }` and some const argument `b` which references the
// first anon const like so: `{ [1; a] }`. The `b` anon const should not be allowed to use
// any generic parameters as:
// - The anon const is not a simple bare parameter, e.g. `N`
// - The anon const is not the *length* of an array repeat expression, e.g. the `N` in `[1; N]`.
//
// On the other hand `a` *is* a const argument for the length of a repeat expression and
// so it *should* inherit the generics declared on its parent definition. (This hack is
// introduced for backwards compatibility and is tracked in #76200)
//
// In this specific case `a`'s parent should be `b` which does not have any generics.
// This means that even though `a` inherits generics from `b`, it still winds up not having
// access to any generic parameters.  If `a` were to inherit its generics from the surrounding
// function `foo` then the reference to `a` from `b` would contain generic parameters not usable
// by `b` which would cause us to ICE.

fn bar<const N: usize>() {}

fn foo<const N: usize>() {
    bar::<{ [1; N] }>();
    //~^ ERROR: generic parameters may not be used in const operations
    bar::<{ [1; { N + 1 }] }>();
    //~^ ERROR: generic parameters may not be used in const operations
}

fn main() {}
