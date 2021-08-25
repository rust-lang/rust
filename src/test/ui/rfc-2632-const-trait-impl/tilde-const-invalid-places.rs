#![feature(const_trait_impl)]
#![feature(associated_type_bounds)]

trait T {}
struct S;
impl T for S {}

fn rpit() -> impl ~const T { S }
//~^ ERROR `~const` is not allowed

fn apit(_: impl ~const T) {}
//~^ ERROR `~const` is not allowed

fn rpit_assoc_bound() -> impl IntoIterator<Item: ~const T> { Some(S) }
//~^ ERROR `~const` is not allowed

fn apit_assoc_bound(_: impl IntoIterator<Item: ~const T>) {}
//~^ ERROR `~const` is not allowed

fn generic<T: ~const T>() {}
//~^ ERROR `~const` is not allowed

fn where_clause<T>() where T: ~const T {}
//~^ ERROR `~const` is not allowed

impl ~const T {}
//~^ ERROR `~const` is not allowed

fn trait_object() -> &'static dyn ~const T { &S }
//~^ ERROR `~const` is not allowed

fn trait_object_in_apit(_: impl IntoIterator<Item = Box<dyn ~const T>>) {}
//~^ ERROR `~const` is not allowed

struct S<T: ~const ?Sized>(std::marker::PhantomData<T>);
//~^ ERROR `~const` and `?` are mutually exclusive

fn main() {}
