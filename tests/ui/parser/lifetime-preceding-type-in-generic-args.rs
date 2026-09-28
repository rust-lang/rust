// Regression test for #59325, specifically for ampersand-less lifetimes in generics.
// Similar tests outside of generics are in recover/recover-ampersand-less-ref-ty.rs.
struct T1;
struct T2;

struct Foo<T>(T);
struct Bar<T1, T2>(T1, T2);

struct A1 {
    x: Foo<'static T1>, //~ERROR expected one of `,` or `>`
    //~^ HELP you might have meant to write a reference type here
    //~| HELP use a comma to separate type parameters
}

struct A2 {
    x: Foo<'static mut T1>, //~ERROR expected one of `,` or `>`
    //~^ HELP you might have meant to write a reference type here
}

struct A3 {
    x: Bar<'static T1, T2>, //~ERROR expected one of `,` or `>`
    //~^ HELP you might have meant to write a reference type here
    //~| HELP use a comma to separate type parameters
}

struct A4 {
    x: Bar<T1, 'static T2>, //~ERROR expected one of `,` or `>`
    //~^ HELP you might have meant to write a reference type here
    //~| HELP use a comma to separate type parameters
}

fn main() {}
