// This test checks that the error messages you get for this example
// at least mention `'a` and `'static`. The precise messages can drift
// over time, but this test used to exhibit some pretty bogus messages
// that were not remotely helpful.

// revisions: base nll
// ignore-compare-mode-nll
//[base] error-pattern:the lifetime `'a`
//[base] error-pattern:the static lifetime
//[nll] compile-flags: -Z borrowck=mir
//[nll] error-pattern:argument requires that `'a` must outlive `'static`

struct Invariant<'a>(Option<&'a mut &'a mut ()>);

fn mk_static() -> Invariant<'static> { Invariant(None) }

fn unify<'a>(x: Option<Invariant<'a>>, f: fn(Invariant<'a>)) {
    let bad = if x.is_some() {
        x.unwrap() //[nll]~ ERROR borrowed data escapes outside of function [E0521]
    } else {
        mk_static() //[base]~ ERROR `if` and `else` have incompatible types [E0308]
    };
    f(bad);
}

fn main() {}
