//@ check-pass

trait Gats<'a> {
    type Assoc;
    type Assoc2;
}

trait Trait: for<'a> Gats<'a> {
    fn foo<'a>(_: &mut <Self as Gats<'a>>::Assoc) -> <Self as Gats<'a>>::Assoc2;
}

impl<'a> Gats<'a> for () {
    type Assoc = &'a u32;
    type Assoc2 = ();
}

type GatsAssoc<'a, T> = <T as Gats<'a>>::Assoc;
type GatsAssoc2<'a, T> = <T as Gats<'a>>::Assoc2;

impl Trait for () {
    fn foo<'a>(_: &mut GatsAssoc<'a, Self>) -> GatsAssoc2<'a, Self> {}
}

fn main() {}
