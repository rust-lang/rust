// Test equality constraints on associated types. Check we get an error when an
// equality constraint is used in an invalid context

struct Bar;
struct Qux;

// Tests for a non generic trait
pub trait Tr1 {
    type A;
    fn boo(&self) -> <Self as Tr1>::A;
}

impl Tr1 for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

// Test for when the assoc type is
// specified as an equality constraint
impl Tr1<A = usize> for usize {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR not all trait items implemented, missing: `A`
    fn boo(&self) -> usize { 42 }
}

// Test for a wronngly used equality constraint in a func arg
fn baz<I: Tr1>(_x: &<I as Tr1<A=Bar>>::A) {}
//~^ ERROR associated item constraints are not allowed here



// Tests for a generic trait
trait Tr2<T1, T2, T3> {
}

// Test for when wrongly specified equality constraint's ident
// matches some generic param's ident
// (Note: E0229 is emitted only for the first erroneous equality
// constraint (T2) not for any subequent ones (e.g. T3))
impl Tr2<i32, T2 = Qux, T3 = usize> for Bar {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR trait takes 3 generic arguments but 1 generic argument was supplied
}

// Test for when equality constraint's ident matches a
// generic param's ident but has different case
impl Tr2<i32, t2 = Qux, T3 = usize> for Qux {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR trait takes 3 generic arguments but 1 generic argument was supplied
}

// Test for when equality constraint's ident
// matches none of the generic param idents
impl Tr2<i32, X = Qux, Y = usize> for Bar {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR trait takes 3 generic arguments but 1 generic argument was supplied
}

// Test for when the term in equality constraint is itself generic
struct GenericTerm<T> { _t: T }
impl Tr2<i32, Qux, T3 = GenericTerm<i32>> for Bar {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR trait takes 3 generic arguments but 2 generic arguments were supplied
}



// Tests for a trait with a const param
trait Tr3<const N: i32, T2, T3> {
}

// Test for when equality constraint's ident
// matches the const param's ident
// (Deliberately spread over multiple lines to test that
// our suggestion spans are kosher in the face of such formatting)
impl Tr3<N
//~^ ERROR associated item constraints are not allowed here
//~| ERROR associated const equality is incomplete
= 42, T2 = Qux, T3 = usize> for Bar {
}

// Test for when equality constraint's ident
// matches the const param's ident but has a different case
impl Tr3<n = 42, T2 = Qux, T3 = usize> for Qux {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR associated const equality is incomplete
//~| ERROR trait takes 3 generic arguments but 0 generic arguments were supplied
}

// Test for when equality constraint's ident
// matches the const param ident but the constraint is a type arg
impl Tr3<N = u32, T2 = Qux, T3 = usize> for Bar {
//~^ ERROR associated item constraints are not allowed here
}

// Test for when equality constraint's ident
// matches a type param ident but the constraint is a const arg
impl Tr3<42, T2 = 42, T3 = usize> for Bar {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR associated const equality is incomplete
//~| ERROR trait takes 3 generic arguments but 1 generic argument was supplied
}

// Test for when equality constraint's ident
// matches none of the param idents
impl Tr3<X = 42, Y = Qux, Z = usize> for Bar {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR associated const equality is incomplete
//~| ERROR trait takes 3 generic arguments but 0 generic arguments were supplied
}



// Test for the case when lifetimes are present
struct St<'a, T> { v: &'a T }

impl<'a, T> St<'a , T = Qux> {
//~^ ERROR associated item constraints are not allowed here
//~| ERROR struct takes 1 generic argument but 0 generic arguments were supplied
}


pub fn main() {}
