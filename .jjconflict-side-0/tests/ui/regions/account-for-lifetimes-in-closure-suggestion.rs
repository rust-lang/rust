// #125634
struct Thing;

// Invariant in 'a, Covariant in 'b
struct TwoThings<'a, 'b>(*mut &'a (), &'b mut ());

impl Thing {
    fn enter_scope<'a>(self, _scope: impl for<'b> FnOnce(TwoThings<'a, 'b>)) {}
}

fn foo() {
    Thing.enter_scope(|ctx| {
        SameLifetime(ctx); //~ ERROR lifetime may not live long enough
    });
}

struct SameLifetime<'a>(TwoThings<'a, 'a>);

fn main() {}
