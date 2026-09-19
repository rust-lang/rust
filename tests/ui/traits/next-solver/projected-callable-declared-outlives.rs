//@ run-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Has<'r> {
    type Assoc: 'r;
}

trait Family {
    type View<'a>;
}

trait Get<'r> {
    type Output: Family;
}

impl<'r, T: Family + 'r> Get<'r> for T {
    type Output = T;
}

fn invoke<'r, 'a, C, T: Family, F>(f: F, value: T::View<'a>) -> T::View<'a>
where
    C: Has<'r, Assoc = T>,
    F: for<'b> Fn(T::View<'b>) -> <<T as Get<'r>>::Output as Family>::View<'b>,
{
    f(value)
}

fn pointer<'r, C, T: Family>()
where
    C: Has<'r, Assoc = T>,
{
    let _: Option<for<'a> fn(T::View<'a>) -> <<T as Get<'r>>::Output as Family>::View<'a>> =
        None;
    let _: for<'a> fn(T::View<'a>) -> T::View<'a> = identity::<'_, 'r, C, T>;
}

fn identity<'a, 'r, C, T: Family>(
    value: T::View<'a>,
) -> <<T as Get<'r>>::Output as Family>::View<'a>
where
    C: Has<'r, Assoc = T>,
    T::View<'a>: Sized,
{
    value
}

fn object<'r, C, T: Family>()
where
    C: Has<'r, Assoc = T>,
{
    let _: Option<
        Box<dyn for<'a> Fn(T::View<'a>) -> <<T as Get<'r>>::Output as Family>::View<'a>>,
    > = None;
}

fn region<'r, 'a, C>(value: &'a ()) -> &'r ()
where
    C: Has<'r, Assoc = &'a ()>,
{
    value
}

fn universal<C, T: Family, F>()
where
    for<'r> C: Has<'r, Assoc = T>,
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait HasGat {
    type Assoc<'a>: 'a;
}

fn universal_gat<C, T: Family, F>()
where
    for<'r> C: HasGat<Assoc<'r> = T>,
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait Conditional {
    type Assoc<'a>: 'a where Self: 'a;
}

fn proven_premise<C: 'static, T: Family, F>()
where
    for<'r> C: Conditional<Assoc<'r> = T>,
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait Required<'r>: 'r {}
trait AllRegions {
    type Assoc: for<'r> Required<'r>;
}

fn quantified_declaration<C: AllRegions<Assoc = T>, T: Family, F>()
where
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait ConditionalOn<'r, T> {
    type Assoc: 'r where T: 'r;
}

fn dependent_premises<C, D, T: Family, U, F>()
where
    for<'r> C: ConditionalOn<'r, U, Assoc = T>,
    D: Has<'static, Assoc = U>,
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait Bound<'a, U>: 'a {}
trait ScopedDeclaration<U> {
    type Assoc: for<'a> Bound<'a, &'a U>;
}

fn proven_declaration_premise<C: ScopedDeclaration<U, Assoc = T>, T: Family, U: 'static, F>()
where
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

struct Borrowed;
impl Family for Borrowed {
    type View<'a> = &'a u32;
}
impl<'r> Has<'r> for Borrowed {
    type Assoc = Self;
}

fn main() {
    let value = 31;
    assert_eq!(*invoke::<'static, '_, Borrowed, Borrowed, _>(|x| x, &value), 31);
    pointer::<'static, Borrowed, Borrowed>();
    object::<'static, Borrowed, Borrowed>();
}
