// This test checks that the error messages you get for this example
// at least mention `'a` and `'static`. The precise messages can drift
// over time, but this test used to exhibit some pretty bogus messages
// that were not remotely helpful.

//@ error-pattern:argument requires that `'a` must outlive `'static`

struct Invariant<'a>(Option<&'a mut &'a mut ()>);

fn mk_static() -> Invariant<'static> { Invariant(None) }

fn unify<'a>(x: Option<Invariant<'a>>, f: fn(Invariant<'a>)) {
    let bad = if x.is_some() {
        x.unwrap() //~ ERROR borrowed data escapes outside of function [E0521]
    } else {
        mk_static()
    };
    f(bad);
}

fn main() {}
