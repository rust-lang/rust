struct S<'a>(&'a u8);
fn foo() {}

// Paren generic args in AnonConst
fn a() -> [u8; foo()] {
    //~^ ERROR mismatched types
    panic!()
}

// Paren generic args in ConstGeneric
fn b<const C: u8()>() {}
//~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

// Paren generic args in AnonymousReportError
fn c<T = u8()>() {}
//~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
//~| ERROR defaults for generic parameters are not allowed here
//~| WARN this was previously accepted

// Elided lifetime in path in ConstGeneric
fn d<const C: S>() {}
//~^ ERROR missing lifetime specifier

trait Foo<'a> {}
struct Bar<const N: &'a (dyn for<'a> Foo<'a>)>;
//~^ ERROR the type of const parameters must not depend on other generic parameters

fn main() {}
