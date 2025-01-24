//@ known-bug: #129372
//@ compile-flags: -Cdebuginfo=2 -Copt-level=0

pub struct Wrapper<T>(T);
struct Struct;

pub trait TraitA {
    type AssocA<'t>;
}
pub trait TraitB {
    type AssocB;
}

pub fn helper(v: impl MethodTrait) {
    let _local_that_causes_ice = v.method();
}

pub fn main() {
    helper(Wrapper(Struct));
}

pub trait MethodTrait {
    type Assoc<'a>;

    fn method(self) -> impl for<'a> FnMut(&'a ()) -> Self::Assoc<'a>;
}

impl<T: TraitB> MethodTrait for T
where
    <T as TraitB>::AssocB: TraitA,
{
    type Assoc<'a> = <T::AssocB as TraitA>::AssocA<'a>;

    fn method(self) -> impl for<'a> FnMut(&'a ()) -> Self::Assoc<'a> {
        move |_| loop {}
    }
}

impl<T, B> TraitB for Wrapper<B>
where
    B: TraitB<AssocB = T>,
{
    type AssocB = T;
}

impl TraitB for Struct {
    type AssocB = Struct;
}

impl TraitA for Struct {
    type AssocA<'t> = Self;
}
