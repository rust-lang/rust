// Test equality constraints on associated types. Check we get an error when an
// equality constraint is used outside of type parameter declarations

pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;
struct Qux;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

fn baz<I: Foo>(_x: &<I as Foo<A=Bar>>::A) {}
//~^ ERROR associated type bindings are not allowed here


trait Tr1<T1> {
}

impl Tr1<T1 = String> for Bar {
//~^ ERROR associated type bindings are not allowed here
//~| ERROR trait takes 1 generic argument but 0 generic arguments were supplied
}


trait Tr2<T1, T2, T3> {
}

// E0229 is emitted only for the first erroneous equality
// constraint (T2) not for any subequent ones (e.g. T3)
impl Tr2<i32, T2 = Qux, T3 = usize> for Bar {
//~^ ERROR associated type bindings are not allowed here
//~| ERROR trait takes 3 generic arguments but 1 generic argument was supplied
}

struct GenericStruct<T> { _t: T }

impl Tr2<i32, Qux, T3 = GenericStruct<i32>> for Bar {
//~^ ERROR associated type bindings are not allowed here
//~| ERROR trait takes 3 generic arguments but 2 generic arguments were supplied
}


// Covers the case when the type has a const param
trait Tr3<const N: i32, T2, T3> {
}

impl Tr3<N = 42, T2 = Qux, T3 = usize> for Bar {
//~^ ERROR associated type bindings are not allowed here
//~| ERROR associated const equality is incomplete
//~| ERROR trait takes 3 generic arguments but 0 generic arguments were supplied
}


// Covers the case when lifetimes
// are present
struct St<'a, T> { v: &'a T }

impl<'a, T> St<'a, T = Qux> {
//~^ ERROR associated type bindings are not allowed here
//~| ERROR struct takes 1 generic argument but 0 generic arguments were supplied
}


// Covers the case when the type
// in question has no generic params
trait Tr4 {
}

impl Tr4<T = Qux, T2 = usize> for Bar {
//~^ ERROR associated type bindings are not allowed here
}


pub fn main() {}
