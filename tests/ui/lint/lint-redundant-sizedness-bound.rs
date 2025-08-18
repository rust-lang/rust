#![deny(redundant_sizedness_bound)]

fn directly<T: Sized + ?Sized>(t: &T) {}
//~^ ERROR redundant_sizedness_bound

trait A: Sized {}
trait B: A {}

fn depth_1<T: A + ?Sized>(t: &T) {}
//~^ ERROR redundant_sizedness_bound
fn depth_2<T: B + ?Sized>(t: &T) {}
//~^ ERROR redundant_sizedness_bound

// We only need to show one
fn multiple_paths<T: A + B + ?Sized>(t: &T) {}
//~^ ERROR redundant_sizedness_bound

fn in_where<T>(t: &T)
where
    T: Sized + ?Sized,
    //~^ ERROR redundant_sizedness_bound
{
}

fn mixed_1<T: Sized>(t: &T)
where
    T: ?Sized,
    //~^ ERROR redundant_sizedness_bound
{
}

fn mixed_2<T: ?Sized>(t: &T)
//~^ ERROR redundant_sizedness_bound
where
    T: Sized,
{
}

fn mixed_3<T>(t: &T)
where
    T: Sized,
    T: ?Sized,
    //~^ ERROR redundant_sizedness_bound
{
}

struct Struct<T: Sized + ?Sized>(T);
//~^ ERROR redundant_sizedness_bound

impl<T: Sized + ?Sized> Struct<T> {
    //~^ ERROR redundant_sizedness_bound
    fn method<U: Sized + ?Sized>(&self) {}
    //~^ ERROR redundant_sizedness_bound
}

enum Enum<T: Sized + ?Sized + 'static> {
    //~^ ERROR redundant_sizedness_bound
    Variant(&'static T),
}

union Union<'a, T: Sized + ?Sized> {
    //~^ ERROR redundant_sizedness_bound
    a: &'a T,
}

trait Trait<T: Sized + ?Sized> {
    //~^ ERROR redundant_sizedness_bound
    fn trait_method<U: Sized + ?Sized>() {}
    //~^ ERROR redundant_sizedness_bound

    type GAT<U: Sized + ?Sized>;
    //~^ ERROR redundant_sizedness_bound

    type Assoc: Sized + ?Sized; // False negative
}

trait SecondInTrait: Send + Sized {}
fn second_in_trait<T: ?Sized + SecondInTrait>() {}
//~^ ERROR redundant_sizedness_bound

fn impl_trait(_: &(impl Sized + ?Sized)) {}
//~^ ERROR redundant_sizedness_bound

trait GenericTrait<T>: Sized {}
fn in_generic_trait<T: GenericTrait<U> + ?Sized, U>() {}
//~^ ERROR redundant_sizedness_bound

mod larger_graph {
    // C1  C2  Sized
    //  \  /\  /
    //   B1  B2
    //    \  /
    //     A1

    trait C1 {}
    trait C2 {}
    trait B1: C1 + C2 {}
    trait B2: C2 + Sized {}
    trait A1: B1 + B2 {}

    fn larger_graph<T: A1 + ?Sized>() {}
    //~^ ERROR redundant_sizedness_bound
}

// Should not lint

fn sized<T: Sized>() {}
fn maybe_sized<T: ?Sized>() {}

struct SeparateBounds<T: ?Sized>(T);
impl<T: Sized> SeparateBounds<T> {}

trait P {}
trait Q: P {}

fn ok_depth_1<T: P + ?Sized>() {}
fn ok_depth_2<T: Q + ?Sized>() {}

//external! {
//    fn in_macro<T: Clone + ?Sized>(t: &T) {}

//    fn with_local_clone<T: $Clone + ?Sized>(t: &T) {}
//}

#[derive(Clone)]
struct InDerive<T: ?Sized> {
    t: T,
}

struct Refined<T: ?Sized>(T);
impl<T: Sized> Refined<T> {}

fn main() {}
