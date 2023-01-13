// revisions: full min

#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

const fn foo<T>() -> usize { std::mem::size_of::<T>() }
const fn bar<const N: usize>() -> usize { N }
const fn faz<'a>(_: &'a ()) -> usize { 13 }
const fn baz<'a>(_: &'a ()) -> usize where &'a (): Sized { 13 }

struct Foo<const N: usize>;
fn test<'a, 'b, T, const N: usize>() where &'b (): Sized {
    let _: [u8; foo::<T>()]; //[min]~ ERROR generic parameters may not
                             //[full]~^ ERROR unconstrained generic constant
    let _: [u8; bar::<N>()]; //[min]~ ERROR generic parameters may not
                             //[min]~^ ERROR unresolved item provided when a constant was expected
                             //[full]~^^ ERROR unconstrained generic constant
    let _: [u8; faz::<'a>(&())]; //[min]~ ERROR a non-static lifetime
                                 //~^ ERROR cannot specify lifetime arguments
    let _: [u8; baz::<'a>(&())]; //[min]~ ERROR a non-static lifetime
    let _: [u8; faz::<'b>(&())]; //[min]~ ERROR a non-static lifetime
                                 //~^ ERROR cannot specify lifetime arguments
    let _: [u8; baz::<'b>(&())]; //[min]~ ERROR a non-static lifetime

    let _ = [0; foo::<T>()]; //[min]~ ERROR constant expression depends on a generic parameter
                             //[full]~^ ERROR unconstrained generic constant
    let _ = [0; bar::<N>()]; //[min]~ ERROR generic parameters may not
                             //[min]~^ ERROR unresolved item provided when a constant was expected
                             //[full]~^^ ERROR unconstrained generic constant
    let _ = [0; faz::<'a>(&())]; //[min]~ ERROR a non-static lifetime
                                 //~^ ERROR cannot specify lifetime arguments
    let _ = [0; baz::<'a>(&())]; //[min]~ ERROR a non-static lifetime
    let _ = [0; faz::<'b>(&())]; //[min]~ ERROR a non-static lifetime
                                 //~^ ERROR cannot specify lifetime arguments
    let _ = [0; baz::<'b>(&())]; //[min]~ ERROR a non-static lifetime
    let _: Foo<{ foo::<T>() }>; //[min]~ ERROR generic parameters may not
                                //[full]~^ ERROR unconstrained generic constant
    let _: Foo<{ bar::<N>() }>; //[min]~ ERROR generic parameters may not
                                //[min]~^ ERROR unresolved item provided when a constant was expected
                                //[full]~^^ ERROR unconstrained generic constant
    let _: Foo<{ faz::<'a>(&()) }>; //[min]~ ERROR a non-static lifetime
                                    //~^ ERROR cannot specify lifetime arguments
    let _: Foo<{ baz::<'a>(&()) }>; //[min]~ ERROR a non-static lifetime
    let _: Foo<{ faz::<'b>(&()) }>; //[min]~ ERROR a non-static lifetime
                                    //~^ ERROR cannot specify lifetime arguments
    let _: Foo<{ baz::<'b>(&()) }>; //[min]~ ERROR a non-static lifetime
    let _ = Foo::<{ foo::<T>() }>; //[min]~ ERROR generic parameters may not
                                   //[full]~^ ERROR unconstrained generic constant
    let _ = Foo::<{ bar::<N>() }>; //[min]~ ERROR generic parameters may not
                                   //[min]~^ ERROR unresolved item provided when a constant was expected
                                   //[full]~^^ ERROR unconstrained generic constant
    let _ = Foo::<{ faz::<'a>(&()) }>; //[min]~ ERROR a non-static lifetime
                                       //~^ ERROR cannot specify lifetime arguments
    let _ = Foo::<{ baz::<'a>(&()) }>; //[min]~ ERROR a non-static lifetime
    let _ = Foo::<{ faz::<'b>(&()) }>; //[min]~ ERROR a non-static lifetime
                                       //~^ ERROR cannot specify lifetime arguments
    let _ = Foo::<{ baz::<'b>(&()) }>; //[min]~ ERROR a non-static lifetime
}

fn main() {}
