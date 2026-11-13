trait Foo {}

trait Trait {
    type Associated;
}
trait DerivedTrait: Trait {}
trait GenericTrait<A> {
    type Associated;
}

struct Impl;
impl Foo for Impl {}
impl Trait for Impl {
    type Associated = ();
}
impl DerivedTrait for Impl {}
impl<A> GenericTrait<A> for Impl {
    type Associated = ();
}

fn returns_opaque() -> impl Trait + 'static {
    Impl
}
fn returns_opaque_derived() -> impl DerivedTrait + 'static {
    Impl
}
fn returns_opaque_foo() -> impl Trait + Foo {
    Impl
}
fn returns_opaque_derived_foo() -> impl DerivedTrait + Foo {
    Impl
}
fn returns_opaque_generic() -> impl GenericTrait<()> + 'static {
    Impl
}
fn returns_opaque_generic_foo() -> impl GenericTrait<()> + Foo {
    Impl
}
fn returns_opaque_generic_duplicate() -> impl GenericTrait<()> + GenericTrait<u8> {
    Impl
}

fn accepts_trait<T: Trait<Associated = ()>>(_: T) {}
fn accepts_generic_trait<T: GenericTrait<(), Associated = ()>>(_: T) {}

fn check_generics<A, B, C, D, E, F, G>(a: A, b: B, c: C, d: D, e: E, f: F, g: G)
where
    A: Trait + 'static,
    B: DerivedTrait + 'static,
    C: Trait + Foo,
    D: DerivedTrait + Foo,
    E: GenericTrait<()> + 'static,
    F: GenericTrait<()> + Foo,
    G: GenericTrait<()> + GenericTrait<u8>,
{
    accepts_trait(a);
    //~^ ERROR type mismatch resolving `<A as Trait>::Associated == ()`

    accepts_trait(b);
    //~^ ERROR type mismatch resolving `<B as Trait>::Associated == ()`

    accepts_trait(c);
    //~^ ERROR type mismatch resolving `<C as Trait>::Associated == ()`

    accepts_trait(d);
    //~^ ERROR type mismatch resolving `<D as Trait>::Associated == ()`

    accepts_generic_trait(e);
    //~^ ERROR type mismatch resolving `<E as GenericTrait<()>>::Associated == ()`

    accepts_generic_trait(f);
    //~^ ERROR type mismatch resolving `<F as GenericTrait<()>>::Associated == ()`

    accepts_generic_trait(g);
    //~^ ERROR type mismatch resolving `<G as GenericTrait<()>>::Associated == ()`
}

fn main() {
    accepts_trait(returns_opaque());
    //~^ ERROR type mismatch resolving `<impl Trait as Trait>::Associated == ()`

    accepts_trait(returns_opaque_derived());
    //~^ ERROR type mismatch resolving `<impl DerivedTrait as Trait>::Associated == ()`

    accepts_trait(returns_opaque_foo());
    //~^ ERROR type mismatch resolving `<impl Trait + Foo as Trait>::Associated == ()`

    accepts_trait(returns_opaque_derived_foo());
    //~^ ERROR type mismatch resolving `<impl DerivedTrait + Foo as Trait>::Associated == ()`

    accepts_generic_trait(returns_opaque_generic());
    //~^ ERROR type mismatch resolving `<impl GenericTrait<()> as GenericTrait<()>>::Associated == ()`

    accepts_generic_trait(returns_opaque_generic_foo());
    //~^ ERROR type mismatch resolving `<impl GenericTrait<()> + Foo as GenericTrait<()>>::Associated == ()`

    accepts_generic_trait(returns_opaque_generic_duplicate());
    //~^ ERROR type mismatch resolving `<impl GenericTrait<()> + GenericTrait<u8> as GenericTrait<()>>::Associated == ()`
}
