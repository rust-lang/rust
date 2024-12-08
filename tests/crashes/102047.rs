//@ known-bug: #102047

struct Ty1;
struct Ty2;

pub trait Trait<T> {}

pub trait WithAssoc1<'a> {
    type Assoc;
}
pub trait WithAssoc2<'a> {
    type Assoc;
}

impl<T, U> Trait<for<'a> fn(<T as WithAssoc1<'a>>::Assoc, <U as WithAssoc2<'a>>::Assoc)> for (T, U)
where
    T: for<'a> WithAssoc1<'a> + for<'a> WithAssoc2<'a, Assoc = i32>,
    U: for<'a> WithAssoc2<'a>,
{
}

impl WithAssoc1<'_> for Ty1 {
    type Assoc = ();
}
impl WithAssoc2<'_> for Ty1 {
    type Assoc = i32;
}
impl WithAssoc1<'_> for Ty2 {
    type Assoc = ();
}
impl WithAssoc2<'_> for Ty2 {
    type Assoc = u32;
}

fn foo<T, U, V>()
where
    T: for<'a> WithAssoc1<'a>,
    U: for<'a> WithAssoc2<'a>,
    (T, U): Trait<V>,
{
}

fn main() {
    foo::<Ty1, Ty2, _>();
}
