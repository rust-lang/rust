// Needed for `type Y = impl Trait<_>` and `type B = _;`
#![feature(associated_type_defaults)]
#![feature(type_alias_impl_trait)]
// This test checks that it is not possible to enable global type
// inference by using the `_` type placeholder.

fn test() -> _ { 5 }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

fn test2() -> (_, _) { (5, 5) }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

static TEST3: _ = "test";
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables

static TEST4: _ = 145;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables

static TEST5: (_, _) = (1, 2);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables

fn test6(_: _) { }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn test6_b<T>(_: _, _: T) { }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn test6_c<T, K, L, A, B>(_: _, _: (T, K, L, A, B)) { }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn test7(x: _) { let _x: usize = x; }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn test8(_f: fn() -> _) { }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
//~^^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

struct Test9;

impl Test9 {
    fn test9(&self) -> _ { () }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

    fn test10(&self, _x : _) { }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
}

fn test11(x: &usize) -> &_ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    &x
}

unsafe fn test12(x: *const usize) -> *const *const _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    &x
}

impl Clone for Test9 {
    fn clone(&self) -> _ { Test9 }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

    fn clone_from(&mut self, other: _) { *self = Test9; }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
}

struct Test10 {
    a: _,
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for structs
    b: (_, _),
}

pub fn main() {
    static A = 42;
    //~^ ERROR missing type for `static` item
    static B: _ = 42;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
    static C: Option<_> = Some(42);
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
    fn fn_test() -> _ { 5 }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

    fn fn_test2() -> (_, _) { (5, 5) }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

    static FN_TEST3: _ = "test";
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables

    static FN_TEST4: _ = 145;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables

    static FN_TEST5: (_, _) = (1, 2);
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables

    fn fn_test6(_: _) { }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

    fn fn_test7(x: _) { let _x: usize = x; }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

    fn fn_test8(_f: fn() -> _) { }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    //~^^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

    struct FnTest9;

    impl FnTest9 {
        fn fn_test9(&self) -> _ { () }
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

        fn fn_test10(&self, _x : _) { }
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    }

    impl Clone for FnTest9 {
        fn clone(&self) -> _ { FnTest9 }
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

        fn clone_from(&mut self, other: _) { *self = FnTest9; }
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    }

    struct FnTest10 {
        a: _,
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for structs
        b: (_, _),
    }

    fn fn_test11(_: _) -> (_, _) { panic!() }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| ERROR type annotations needed

    fn fn_test12(x: i32) -> (_, _) { (x, x) }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

    fn fn_test13(x: _) -> (i32, _) { (x, x) }
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
}

trait T {
    fn method_test1(&self, x: _);
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    fn method_test2(&self, x: _) -> _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    fn method_test3(&self) -> _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    fn assoc_fn_test1(x: _);
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    fn assoc_fn_test2(x: _) -> _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
    fn assoc_fn_test3() -> _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
}

struct BadStruct<_>(_);
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR the placeholder `_` is not allowed within types on item signatures for structs
trait BadTrait<_> {}
//~^ ERROR expected identifier, found reserved identifier `_`
impl BadTrait<_> for BadStruct<_> {}
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for implementations

fn impl_trait() -> impl BadTrait<_> {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for opaque types
    unimplemented!()
}

struct BadStruct1<_, _>(_);
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR expected identifier, found reserved identifier `_`
//~| ERROR the name `_` is already used
//~| ERROR the placeholder `_` is not allowed within types on item signatures for structs
struct BadStruct2<_, T>(_, T);
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR the placeholder `_` is not allowed within types on item signatures for structs

type X = Box<_>;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for type aliases

struct Struct;
trait Trait<T> {}
impl Trait<usize> for Struct {}
type Y = impl Trait<_>;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for opaque types
fn foo() -> Y {
    Struct
}

trait Qux {
    type A;
    type B = _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
    const C: _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
    const D: _ = 42;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
    // type E: _; // FIXME: make the parser propagate the existence of `B`
    type F: std::ops::Fn(_);
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
}
impl Qux for Struct {
    type A = _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
    type B = _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
    const C: _;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
    //~| ERROR associated constant in `impl` without body
    const D: _ = 42;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
}

fn map<T>(_: fn() -> Option<&'static T>) -> Option<T> {
    None
}

fn value() -> Option<&'static _> {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    Option::<&'static u8>::None
}

const _: Option<_> = map(value);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants

fn evens_squared(n: usize) -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    (1..n).filter(|x| x % 2 == 0).map(|x| x * x)
}

const _: _ = (1..10).filter(|x| x % 2 == 0).map(|x| x * x);
//~^ ERROR the trait bound
//~| ERROR the trait bound
//~| ERROR the placeholder
