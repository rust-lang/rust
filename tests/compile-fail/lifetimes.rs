#![feature(plugin)]
#![plugin(clippy)]

#![deny(needless_lifetimes)]
#![allow(dead_code)]
fn distinct_lifetimes<'a, 'b>(_x: &'a u8, _y: &'b u8, _z: u8) { }
//~^ERROR explicit lifetimes given

fn distinct_and_static<'a, 'b>(_x: &'a u8, _y: &'b u8, _z: &'static u8) { }
//~^ERROR explicit lifetimes given

fn same_lifetime_on_input<'a>(_x: &'a u8, _y: &'a u8) { } // no error, same lifetime on two params

fn only_static_on_input(_x: &u8, _y: &u8, _z: &'static u8) { } // no error, static involved

fn mut_and_static_input(_x: &mut u8, _y: &'static str) { }

fn in_and_out<'a>(x: &'a u8, _y: u8) -> &'a u8 { x }
//~^ERROR explicit lifetimes given

fn multiple_in_and_out_1<'a>(x: &'a u8, _y: &'a u8) -> &'a u8 { x } // no error, multiple input refs

fn multiple_in_and_out_2<'a, 'b>(x: &'a u8, _y: &'b u8) -> &'a u8 { x } // no error, multiple input refs

fn in_static_and_out<'a>(x: &'a u8, _y: &'static u8) -> &'a u8 { x } // no error, static involved

fn deep_reference_1<'a, 'b>(x: &'a u8, _y: &'b u8) -> Result<&'a u8, ()> { Ok(x) } // no error

fn deep_reference_2<'a>(x: Result<&'a u8, &'a u8>) -> &'a u8 { x.unwrap() } // no error, two input refs

fn deep_reference_3<'a>(x: &'a u8, _y: u8) -> Result<&'a u8, ()> { Ok(x) }
//~^ERROR explicit lifetimes given

// where clause, but without lifetimes
fn where_clause_without_lt<'a, T>(x: &'a u8, _y: u8) -> Result<&'a u8, ()> where T: Copy { Ok(x) }
//~^ERROR explicit lifetimes given

type Ref<'r> = &'r u8;

fn lifetime_param_1<'a>(_x: Ref<'a>, _y: &'a u8) { } // no error, same lifetime on two params

fn lifetime_param_2<'a, 'b>(_x: Ref<'a>, _y: &'b u8) { }
//~^ERROR explicit lifetimes given

fn lifetime_param_3<'a, 'b: 'a>(_x: Ref<'a>, _y: &'b u8) { } // no error, bounded lifetime

fn lifetime_param_4<'a, 'b>(_x: Ref<'a>, _y: &'b u8) where 'b: 'a { } // no error, bounded lifetime

struct Lt<'a, I: 'static> {
    x: &'a I
}

fn fn_bound<'a, F, I>(_m: Lt<'a, I>, _f: F) -> Lt<'a, I>
    where F: Fn(Lt<'a, I>) -> Lt<'a, I>  // no error, fn bound references 'a
{ unreachable!() }

fn fn_bound_2<'a, F, I>(_m: Lt<'a, I>, _f: F) -> Lt<'a, I>  //~ERROR explicit lifetimes given
    where for<'x> F: Fn(Lt<'x, I>) -> Lt<'x, I>
{ unreachable!() }

struct X {
    x: u8,
}

impl X {
    fn self_and_out<'s>(&'s self) -> &'s u8 { &self.x }
    //~^ERROR explicit lifetimes given

    fn self_and_in_out<'s, 't>(&'s self, _x: &'t u8) -> &'s u8 { &self.x } // no error, multiple input refs

    fn distinct_self_and_in<'s, 't>(&'s self, _x: &'t u8) { }
    //~^ERROR explicit lifetimes given

    fn self_and_same_in<'s>(&'s self, _x: &'s u8) { } // no error, same lifetimes on two params
}

struct Foo<'a>(&'a u8);

impl<'a> Foo<'a> {
    fn self_shared_lifetime(&self, _: &'a u8) {} // no error, lifetime 'a not defined in method
    fn self_bound_lifetime<'b: 'a>(&self, _: &'b u8) {} // no error, bounds exist
}

fn already_elided<'a>(_: &u8, _: &'a u8) -> &'a u8 {
    unimplemented!()
}

fn struct_with_lt<'a>(_foo: Foo<'a>) -> &'a str { unimplemented!() } //~ERROR explicit lifetimes given

// no warning, two input lifetimes (named on the reference, anonymous on Foo)
fn struct_with_lt2<'a>(_foo: &'a Foo) -> &'a str { unimplemented!() }

// no warning, two input lifetimes (anonymous on the reference, named on Foo)
fn struct_with_lt3<'a>(_foo: &Foo<'a> ) -> &'a str { unimplemented!() }

// no warning, two input lifetimes
fn struct_with_lt4<'a, 'b>(_foo: &'a Foo<'b> ) -> &'a str { unimplemented!() }

trait WithLifetime<'a> {}
type WithLifetimeAlias<'a> = WithLifetime<'a>;

// should not warn because it won't build without the lifetime
fn trait_obj_elided<'a>(_arg: &'a WithLifetime) -> &'a str { unimplemented!() }

// this should warn because there is no lifetime on Drop, so this would be
// unambiguous if we elided the lifetime
fn trait_obj_elided2<'a>(_arg: &'a Drop) -> &'a str { unimplemented!() } //~ERROR explicit lifetimes given

type FooAlias<'a> = Foo<'a>;

fn alias_with_lt<'a>(_foo: FooAlias<'a>) -> &'a str { unimplemented!() } //~ERROR explicit lifetimes given

// no warning, two input lifetimes (named on the reference, anonymous on Foo)
fn alias_with_lt2<'a>(_foo: &'a FooAlias) -> &'a str { unimplemented!() }

// no warning, two input lifetimes (anonymous on the reference, named on Foo)
fn alias_with_lt3<'a>(_foo: &FooAlias<'a> ) -> &'a str { unimplemented!() }

// no warning, two input lifetimes
fn alias_with_lt4<'a, 'b>(_foo: &'a FooAlias<'b> ) -> &'a str { unimplemented!() }

fn main() {
}
