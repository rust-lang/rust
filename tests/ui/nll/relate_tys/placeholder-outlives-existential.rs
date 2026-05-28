// Test that we correctly handle some cases of placeholder leaks.
//
//@ compile-flags:-Zno-leak-check


struct Co<'a>(&'a ());
struct Inv<'a>(*mut &'a ());
struct Contra<'a>(fn(&'a ()));

// `exists<'e> forall<'p> 'p: 'e` -> ERROR
fn p_outlives_e(
    x: for<'e> fn(for<'p> fn(fn(fn(Contra<'e>, Co<'p>)))),
) -> fn(fn(fn(for<'unify> fn(Contra<'unify>, Co<'unify>)))) {
    x //~ ERROR mismatched types [E0308]
}

// `exists<'e> forall<'p> 'e: 'p` -> Ok, 'e: 'static
fn e_outlives_p_static(
    x: for<'e> fn(Inv<'e>, for<'p> fn(fn(fn(Contra<'p>, Co<'e>)))),
) -> fn(Inv<'static>, fn(fn(for<'unify> fn(Contra<'unify>, Co<'unify>)))) {
    x
}

// `exists<'e> forall<'p> 'e: 'p` -> Ok, 'e: 'static -> ERROR
fn e_outlives_p_static_err<'not_static>(
    x: for<'e> fn(Inv<'e>, for<'p> fn(fn(fn(Contra<'p>, Co<'e>)))),
) -> fn(Inv<'not_static>, fn(fn(for<'unify> fn(Contra<'unify>, Co<'unify>)))) {
    x //~ ERROR lifetime may not live long enough
}

fn main() {}
