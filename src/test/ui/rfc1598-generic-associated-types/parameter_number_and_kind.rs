#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete
#![feature(associated_type_defaults)]

// FIXME(#44265): "lifetime parameters are not allowed on this type" errors will be addressed in a
// follow-up PR.

// FIXME(#44265): Update expected errors once E110 is resolved, now does not get past `trait Foo`.

trait Foo {
    type A<'a>;
    type B<'a, 'b>;
    type C;
    type D<T>;
    type E<'a, T>;
    // Test parameters in default values
    type FOk<T> = Self::E<'static, T>;
    //~^ ERROR type parameters are not allowed on this type [E0109]
    //~| ERROR lifetime parameters are not allowed on this type [E0110]
    type FErr1 = Self::E<'static, 'static>; // Error
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
    type FErr2<T> = Self::E<'static, T, u32>; // Error
    //~^ ERROR type parameters are not allowed on this type [E0109]
    //~| ERROR lifetime parameters are not allowed on this type [E0110]
}

struct Fooy;

impl Foo for Fooy {
    type A = u32; // Error: parameter expected
    type B<'a, T> = Vec<T>; // Error: lifetime param expected
    type C<'a> = u32; // Error: no param expected
    type D<'a> = u32; // Error: type param expected
    type E<T, U> = u32; // Error: lifetime expected as the first param
}

struct Fooer;

impl Foo for Fooer {
    type A<T> = u32; // Error: lifetime parameter expected
    type B<'a> = u32; // Error: another lifetime param expected
    type C<T> = T; // Error: no param expected
    type D<'b, T> = u32; // Error: unexpected lifetime param
    type E<'a, 'b> = u32; // Error: type expected as the second param
}

fn main() {}
