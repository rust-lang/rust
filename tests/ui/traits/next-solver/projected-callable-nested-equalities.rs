//@ run-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Bound<'r> {
    type Out: 'r;
}

trait Has {
    type Assoc: for<'r> Bound<'r, Out = Self::Assoc>;
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

fn invoke<'a, C: Has<Assoc = T>, T: Family, F>(f: F, value: T::View<'a>) -> T::View<'a>
where
    F: for<'b> Fn(T::View<'b>) -> <<T as Get<'static>>::Output as Family>::View<'b>,
{
    f(value)
}

fn identity<'a, C: Has<Assoc = T>, T: Family>(
    value: T::View<'a>,
) -> <<T as Get<'static>>::Output as Family>::View<'a> {
    value
}

fn pointer<C: Has<Assoc = T>, T: Family>() {
    let _: for<'a> fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a> =
        identity::<C, T>;
}

fn object<C: Has<Assoc = T>, T: Family>() {
    let _: Option<Box<
        dyn for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
    >> = None;
}

fn invoke_object<'a, C: Has<Assoc = T>, T: Family>(
    f: &dyn for<'b> Fn(T::View<'b>) -> <<T as Get<'static>>::Output as Family>::View<'b>,
    value: T::View<'a>,
) -> T::View<'a> {
    f(value)
}

trait Middle<'r> {
    type Next: Bound<'r, Out = Self::Next>;
}

trait Nested<'r> {
    type Assoc: Middle<'r, Next = Self::Assoc>;
}

fn multiple<'r, C: Nested<'r, Assoc = T>, T: Family, F>()
where
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'r>>::Output as Family>::View<'a>,
{
}

trait Other<U> {
    type Assoc: for<'r> Bound<'r, Out = U>;
}

fn other<C: Other<U, Assoc = T>, T, U: Family, F>()
where
    F: for<'a> Fn(U::View<'a>) -> <<U as Get<'static>>::Output as Family>::View<'a>,
{
}

trait ScopedBound<'r, U> {
    type Out: 'r;
}

trait Scoped<U> {
    type Assoc: for<'r> ScopedBound<'r, &'r U, Out = Self::Assoc>;
}

fn proven_scope<C: Scoped<U, Assoc = T>, T: Family, U: 'static, F>()
where
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait ConditionalBound<'r, U> {
    type Out: 'r where U: 'r;
}

trait Conditional<U> {
    type Assoc: for<'r> ConditionalBound<'r, U, Out = Self::Assoc>;
}

fn seeded_cycle<C: Conditional<U, Assoc = T>, D: Conditional<T, Assoc = U>, T, U>()
where
    T: 'static,
{
    fn require_static<T: 'static>() {}
    require_static::<U>();
}

trait Recur {
    type Out: 'static + Recur<Out = Vec<Self::Out>>;
}

impl<T: 'static> Recur for Vec<T> {
    type Out = Vec<Vec<T>>;
}

fn recursive<C: Recur<Out = T>, T: Family, F>()
where
    F: for<'a> Fn(T::View<'a>) -> <<T as Get<'static>>::Output as Family>::View<'a>,
{
}

trait Unbounded {
    type Out: Unbounded<Out = Vec<Self::Out>>;
}

impl<T> Unbounded for Vec<T> {
    type Out = Vec<Vec<T>>;
}

fn irrelevant<C: Unbounded<Out = T>, T>() {}

fn alternatives<'a, 'b: 'r, 'r, C: Nested<'a, Assoc = T>, D: Nested<'b, Assoc = T>, T>(
    value: T,
) -> impl FnOnce() -> Box<dyn std::fmt::Debug + 'r>
where
    T: std::fmt::Debug,
{
    move || Box::new(value)
}

trait Reader {
    type Offset;
}

struct Unit<R: Reader> {
    reader: R,
    offset: R::Offset,
}

fn closure<'a, R: Reader>(x: &'a Unit<R>) -> impl FnOnce() -> &'a R::Offset {
    let map = move |r: &'a Unit<R>| &r.offset;
    let complete = |r| Some(map(r));
    let _ = complete(x);
    move || map(x)
}

trait Good<'r>: Bound<'r, Out = Self> {}
trait ScopedPath<'r, U>: Good<'r> {}
trait MultiplePaths<U> {
    type Assoc: for<'r> ScopedPath<'r, &'r U> + for<'r> Good<'r>;
}

fn independent_supertrait<C: MultiplePaths<U, Assoc = T>, T, U>(
    value: T,
) -> Box<dyn std::any::Any> {
    Box::new(value)
}

struct Borrowed;

impl Family for Borrowed {
    type View<'a> = &'a u32;
}

impl Has for Borrowed {
    type Assoc = Self;
}

impl<'r> Bound<'r> for Borrowed {
    type Out = Self;
}

fn main() {
    let value = 37;
    assert_eq!(*invoke::<Borrowed, Borrowed, _>(identity::<Borrowed, Borrowed>, &value), 37);
    pointer::<Borrowed, Borrowed>();
    object::<Borrowed, Borrowed>();
    assert_eq!(
        *invoke_object::<Borrowed, Borrowed>(&identity::<Borrowed, Borrowed>, &value),
        37,
    );
    irrelevant::<Vec<()>, Vec<Vec<()>>>();
}
