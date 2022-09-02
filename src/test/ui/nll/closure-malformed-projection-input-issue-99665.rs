// Regression test for #99665
//
// Here we are generating region constraints
// when normalizing input types of the closure.

// check-fail

pub trait MyComponent {
    type Properties;
}

struct Ty1<T>(T);
struct Ty2<T>(T);

impl<M> MyComponent for Ty1<M>
where
    M: 'static,
{
    type Properties = ();
}

impl<M> MyComponent for Ty2<M>
where
    M: 'static,
{
    type Properties = &'static M;
}

fn fail() {
    // This should fail because `Ty1<&u8>` is inferred to be higher-ranked.
    // So effectively we're trying to prove `for<'a> Ty1<&'a u8>: MyComponent`.
    |_: <Ty1<&u8> as MyComponent>::Properties| {};
    //~^ ERROR lifetime may not live long enough
    //~| ERROR higher-ranked subtype error
    //~| ERROR higher-ranked lifetime error
    //~| ERROR higher-ranked lifetime error
    //~| ERROR higher-ranked lifetime error

    |_: <Ty2<&u8> as MyComponent>::Properties| {};
    //~^ ERROR higher-ranked subtype error
    //~| ERROR higher-ranked lifetime error
    //~| ERROR higher-ranked lifetime error
    //~| ERROR higher-ranked lifetime error
}

fn pass() {
    // Here both are not higher-ranked, so they sould pass.
    || -> <Ty1<&u8> as MyComponent>::Properties { panic!() };
    || -> <Ty2<&u8> as MyComponent>::Properties { panic!() };
}

fn main() {}
