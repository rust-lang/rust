//@ compile-flags: -Znext-solver
//@ check-pass

trait Trait {
    type Assoc: Sized;
}

impl Trait for &'static str {
    type Assoc = &'static str;
}

// Wrapper is just here to get around stupid `Sized` obligations in mir typeck
struct Wrapper<T: ?Sized>(std::marker::PhantomData<T>);
fn mk<T: Trait>(x: T) -> Wrapper<<T as Trait>::Assoc> { todo!() }


trait IsStaticStr {}
impl IsStaticStr for (&'static str,) {}
fn define<T: IsStaticStr>(_: T) {}

fn foo<'a, T: Trait>() {
    let y = Default::default();

    // `<?0 as Trait>::Assoc <: &'a str`
    // In the old solver, this would *equate* the LHS and RHS.
    let _: Wrapper<&'a str> = mk(y);

    // ... then later on, we constrain `?0 = &'static str`
    // but that should not mean that `'a = 'static`, because
    // we should use *sub* above.
    define((y,));
}

fn main() {}
