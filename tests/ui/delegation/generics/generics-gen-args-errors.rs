//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod test_1 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}
    reuse foo as bar;

    fn check<A, B, C>() {
        bar::<1, 2, 3, 4, 5, 6>();
        //~^ ERROR: function takes 3 generic arguments but 6 generic arguments were supplied

        bar::<String, String, { String }>();
        //~^ ERROR: expected value, found struct `String` [E0423]

        bar::<'static, 'static, 'static, 'static, 'static>();
        //~^ ERROR: function takes 2 lifetime arguments but 5 lifetime arguments were supplied

        bar::<String, 1, 'static, i32, 'static>();
        //~^ ERROR: constant provided when a type was expected

        bar();

        bar::<_, _, _, _, _>();
        //~^ ERROR: function takes 3 generic arguments but 5 generic arguments were supplied

        bar::<asd, asd, asd>();
        //~^ ERROR: cannot find type `asd` in this scope
        //~| ERROR: cannot find type `asd` in this scope
        //~| ERROR: cannot find type `asd` in this scope
        //~| ERROR: unresolved item provided when a constant was expected

        reuse foo::<A, B, C> as xd;
        //~^ ERROR can't use generic parameters from outer item
        //~| ERROR can't use generic parameters from outer item
        //~| ERROR can't use generic parameters from outer item
        //~| ERROR: unresolved item provided when a constant was expected
    }
}

mod test_2 {
    fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

    reuse foo::<> as bar1;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions

    reuse foo::<String, String> as bar2;
    //~^ ERROR: function takes 3 generic arguments but 2 generic arguments were supplied

    reuse foo::<'static, _, 'asdasd, 'static, 'static, 'static, _> as bar3;
    //~^ ERROR: use of undeclared lifetime name `'asdasd`
    //~| ERROR: function takes 2 lifetime arguments but 5 lifetime arguments were supplied
    //~| ERROR: function takes 3 generic arguments but 2 generic arguments were supplied
    reuse foo::<String, 'static, 123, asdasd> as bar4;
    //~^ ERROR: cannot find type `asdasd` in this scope
    //~| ERROR: function takes 2 lifetime arguments but 1 lifetime argument was supplied

    reuse foo::<1, 2, _, 4, 5, _> as bar5;
    //~^ ERROR: function takes 3 generic arguments but 6 generic arguments were supplied

    reuse foo::<1, 2,asd,String, { let x = 0; }> as bar6;
    //~^ ERROR: cannot find type `asd` in this scope
    //~| ERROR: function takes 3 generic arguments but 5 generic arguments were supplied

    reuse foo::<"asdasd", asd, "askdn", 'static, 'a> as bar7;
    //~^ ERROR: use of undeclared lifetime name `'a`
    //~| ERROR: cannot find type `asd` in this scope
    //~| ERROR: constant provided when a type was expected

    reuse foo::<{}, {}, {}> as bar8;
    //~^ ERROR: constant provided when a type was expected
}

mod test_3 {
    trait Trait<'b, 'c, 'a, T, const N: usize>: Sized {
        fn foo<'d: 'd, U, const M: bool>(self) {}
    }

    reuse Trait::<asd, asd, asd, asd, asd, asdasa>::foo as bar1;
    //~^ ERROR: cannot find type `asd` in this scope
    //~| ERROR: cannot find type `asd` in this scope
    //~| ERROR: cannot find type `asd` in this scope
    //~| ERROR: cannot find type `asd` in this scope
    //~| ERROR: cannot find type `asd` in this scope
    //~| ERROR: cannot find type `asdasa` in this scope
    //~| ERROR: trait takes 3 lifetime arguments but 0 lifetime arguments were supplied
    //~| ERROR: trait takes 2 generic arguments but 6 generic arguments were supplied

    reuse Trait::<'static, 'static>::foo as bar2;
    //~^ ERROR: trait takes 3 lifetime arguments but 2 lifetime arguments were supplied
    //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions
    reuse Trait::<1, 2, 3, 4, 5>::foo as bar3;
    //~^ ERROR: trait takes 3 lifetime arguments but 0 lifetime arguments were supplied
    //~| ERROR: trait takes 2 generic arguments but 5 generic arguments were supplied

    reuse Trait::<1, 2, true>::foo as bar4;
    //~^ ERROR: trait takes 3 lifetime arguments but 0 lifetime arguments were supplied
    //~| ERROR: trait takes 2 generic arguments but 3 generic arguments were supplied

    reuse Trait::<'static>::foo as bar5;
    //~^ ERROR: trait takes 3 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions

    reuse Trait::<1, 2, 'static, DDDD>::foo::<1, 2, 3, 4, 5, 6> as bar6;
    //~^ ERROR: cannot find type `DDDD` in this scope [E0425]
    //~| ERROR: trait takes 3 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR: trait takes 2 generic arguments but 3 generic arguments were supplied
    //~| ERROR: method takes 2 generic arguments but 6 generic arguments were supplied

    reuse Trait::<Trait, Clone, _, 'static, dyn Send, _>::foo::<1, 2, 3, _, 6> as bar7;
    //~^ ERROR: missing lifetime specifiers [E0106]
    //~| ERROR: trait takes 3 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR: trait takes 2 generic arguments but 5 generic arguments were supplied
    //~| ERROR: method takes 2 generic arguments but 5 generic arguments were supplied
}

fn main() {}
