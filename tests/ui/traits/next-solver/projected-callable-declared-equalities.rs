//@ run-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Family {
    type View<'a>;
}

trait Identity {
    type Output: Family;
}

trait Carrier {
    type Assoc: Identity<Output = Self::Assoc>;
}

fn identity<'a, C: Carrier<Assoc = T>, T: Family>(
    value: T::View<'a>,
) -> <<T as Identity>::Output as Family>::View<'a> {
    value
}

fn invoke<'a, C: Carrier<Assoc = T>, T: Family, F>(
    f: F,
    value: T::View<'a>,
) -> T::View<'a>
where
    F: for<'b> Fn(T::View<'b>) -> <<T as Identity>::Output as Family>::View<'b>,
{
    f(value)
}

fn pointer<C: Carrier<Assoc = T>, T: Family>() {
    let _: for<'a> fn(T::View<'a>) -> T::View<'a> = identity::<C, T>;
}

fn object<'a, C: Carrier<Assoc = T>, T: Family>(
    f: Box<dyn for<'b> Fn(T::View<'b>) -> <<T as Identity>::Output as Family>::View<'b>>,
    value: T::View<'a>,
) -> T::View<'a> {
    f(value)
}

trait Middle {
    type Next: Identity<Output = Self::Next>;
}

trait Nested {
    type Assoc: Middle<Next = Self::Assoc>;
}

fn nested<'a, C: Nested<Assoc = T>, T: Family>(
    value: T::View<'a>,
) -> <<T as Identity>::Output as Family>::View<'a> {
    value
}

trait Other<U> {
    type Assoc: Identity<Output = U>;
}

fn other<'a, C: Other<U, Assoc = T>, T, U: Family>(
    value: U::View<'a>,
) -> <<T as Identity>::Output as Family>::View<'a> {
    value
}

trait Gat {
    type Assoc<'a>: Identity<Output = Self::Assoc<'a>>;
}

fn quantified<'a, C, T: Family>(
    value: T::View<'a>,
) -> <<T as Identity>::Output as Family>::View<'a>
where
    for<'b> C: Gat<Assoc<'b> = T>,
{
    value
}

trait ForRegion<'a> {
    type Output;
}

trait ScopedIdentity<'a, U>: ForRegion<'a, Output = Self> {}

trait ScopedCarrier<U> {
    type Assoc: for<'a> ScopedIdentity<'a, &'a U>;
}

fn alternatives<C, D, T, U: 'static, V>(value: T) -> <T as ForRegion<'static>>::Output
where
    C: ScopedCarrier<U, Assoc = T>,
    D: ScopedCarrier<V, Assoc = T>,
{
    value
}

fn reversed_alternatives<C, D, T, U, V: 'static>(value: T) -> <T as ForRegion<'static>>::Output
where
    C: ScopedCarrier<U, Assoc = T>,
    D: ScopedCarrier<V, Assoc = T>,
{
    value
}

struct Borrowed;

impl Family for Borrowed {
    type View<'a> = &'a u32;
}

impl Identity for Borrowed {
    type Output = Self;
}

impl Carrier for Borrowed {
    type Assoc = Self;
}

impl Middle for Borrowed {
    type Next = Self;
}

impl Nested for Borrowed {
    type Assoc = Self;
}

impl Other<Self> for Borrowed {
    type Assoc = Self;
}

impl Gat for Borrowed {
    type Assoc<'a> = Self;
}

fn main() {
    let value = 41;
    assert_eq!(*invoke::<Borrowed, Borrowed, _>(identity::<Borrowed, Borrowed>, &value), 41);
    assert_eq!(*nested::<Borrowed, Borrowed>(&value), 41);
    assert_eq!(*other::<Borrowed, Borrowed, Borrowed>(&value), 41);
    assert_eq!(*quantified::<Borrowed, Borrowed>(&value), 41);
    pointer::<Borrowed, Borrowed>();
    assert_eq!(*object::<Borrowed, Borrowed>(Box::new(identity::<Borrowed, Borrowed>), &value), 41);
}
