// This test checks that the error messages you get for this example
// at least mention `'a` and `'static`. The precise messages can drift
// over time, but this test used to exhibit some pretty bogus messages
// that were not remotely helpful.

//@ dont-require-annotations: NOTE

struct Invariant<'a>(Option<&'a mut &'a mut ()>);

fn mk_static() -> Invariant<'static> { Invariant(None) }

fn unify<'a>(x: Option<Invariant<'a>>, f: fn(Invariant<'a>)) {
    let bad = if x.is_some() {
        x.unwrap()
    } else {
        mk_static() //~ ERROR lifetime may not live long enough
                    //~| NOTE assignment requires that `'a` must outlive `'static`
    };
    f(bad);
}

fn main() {}
