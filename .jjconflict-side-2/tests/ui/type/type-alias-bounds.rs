// Test `ignored_generic_bounds` lint warning about bounds in type aliases.

//@ check-pass
#![allow(dead_code)]

use std::rc::Rc;

type SVec<T: Send + Send> = Vec<T>;
//~^ WARN bounds on generic parameters in type aliases are not enforced [type_alias_bounds]
type S2Vec<T> where T: Send = Vec<T>;
//~^ WARN where clauses on type aliases are not enforced [type_alias_bounds]
type VVec<'b, 'a: 'b + 'b> = (&'b u32, Vec<&'a i32>);
//~^ WARN bounds on generic parameters in type aliases are not enforced [type_alias_bounds]
type WVec<'b, T: 'b + 'b> = (&'b u32, Vec<T>);
//~^ WARN bounds on generic parameters in type aliases are not enforced [type_alias_bounds]
type W2Vec<'b, T> where T: 'b, T: 'b = (&'b u32, Vec<T>);
//~^ WARN where clauses on type aliases are not enforced [type_alias_bounds]

static STATIC: u32 = 0;

fn foo<'a>(y: &'a i32) {
    // If any of the bounds above would matter, the code below would be rejected.
    // This can be seen when replacing the type aliases above by newtype structs.
    // (The type aliases have no unused parameters to make that a valid transformation.)
    let mut x: SVec<_> = Vec::new();
    x.push(Rc::new(42)); // is not send

    let mut x: S2Vec<_> = Vec::new();
    x.push(Rc::new(42)); // is not `Send`

    let mut x: VVec<'static, 'a> = (&STATIC, Vec::new());
    x.1.push(y); // `'a: 'static` does not hold

    let mut x: WVec<'static, &'a i32> = (&STATIC, Vec::new());
    x.1.push(y); // `&'a i32: 'static` does not hold

    let mut x: W2Vec<'static, &'a i32> = (&STATIC, Vec::new());
    x.1.push(y); // `&'a i32: 'static` does not hold
}

// Bounds are not checked either; i.e., the definition is not necessarily well-formed.
struct Sendable<T: Send>(T);
type MySendable<T> = Sendable<T>; // no error here!

// Bounds on type params do enable shorthand type alias paths.
// However, that doesn't actually mean that they are properly enforced.
trait Bound { type Assoc; }
type T1<U: Bound> = U::Assoc; //~ WARN are not enforced
type T2<U> where U: Bound = U::Assoc;  //~ WARN are not enforced

// This errors:
// `type T3<U> = U::Assoc;`
// Do this instead:
type T4<U> = <U as Bound>::Assoc;

// Make sure the help about associated types is not shown incorrectly
type T5<U: Bound> = <U as Bound>::Assoc;  //~ WARN are not enforced
type T6<U: Bound> = ::std::vec::Vec<U>;  //~ WARN are not enforced

fn main() {}
