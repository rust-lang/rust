#![feature(type_alias_impl_trait)] // Needed for single test `type Y = impl Trait<_>`
// This test checks that it is not possible to enable global type
// inference by using the `_` type placeholder.

fn test() -> _ { 5 }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

fn test2() -> (_, _) { (5, 5) }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

static TEST3: _ = "test";
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

static TEST4: _ = 145;
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

static TEST5: (_, _) = (1, 2);
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

fn test6(_: _) { }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

fn test6_b<T>(_: _, _: T) { }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

fn test6_c<T, K, L, A, B>(_: _, _: (T, K, L, A, B)) { }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

fn test7(x: _) { let _x: usize = x; }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

fn test8(_f: fn() -> _) { }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

struct Test9;

impl Test9 {
    fn test9(&self) -> _ { () }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    fn test10(&self, _x : _) { }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
}

fn test11(x: &usize) -> &_ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    &x
}

unsafe fn test12(x: *const usize) -> *const *const _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    &x
}

impl Clone for Test9 {
    fn clone(&self) -> _ { Test9 }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    fn clone_from(&mut self, other: _) { *self = Test9; }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
}

struct Test10 {
    a: _,
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    b: (_, _),
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
}

pub fn main() {
    static A = 42;
    //~^ ERROR missing type for `static` item
    static B: _ = 42;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    static C: Option<_> = Some(42);
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    fn fn_test() -> _ { 5 }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    fn fn_test2() -> (_, _) { (5, 5) }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    static FN_TEST3: _ = "test";
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    static FN_TEST4: _ = 145;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    static FN_TEST5: (_, _) = (1, 2);
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    fn fn_test6(_: _) { }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures

    fn fn_test7(x: _) { let _x: usize = x; }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures

    fn fn_test8(_f: fn() -> _) { }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures

    struct FnTest9;

    impl FnTest9 {
        fn fn_test9(&self) -> _ { () }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

        fn fn_test10(&self, _x : _) { }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
        //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    }

    impl Clone for FnTest9 {
        fn clone(&self) -> _ { FnTest9 }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

        fn clone_from(&mut self, other: _) { *self = FnTest9; }
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
        //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    }

    struct FnTest10 {
        a: _,
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
        //~| ERROR the type placeholder `_` is not allowed within types on item signatures
        b: (_, _),
        //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
        //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    }

    fn fn_test11(_: _) -> (_, _) { panic!() }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR type annotations needed

    fn fn_test12(x: i32) -> (_, _) { (x, x) }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures

    fn fn_test13(x: _) -> (i32, _) { (x, x) }
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
}

trait T {
    fn method_test1(&self, x: _);
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    fn method_test2(&self, x: _) -> _;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    fn method_test3(&self) -> _;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    fn assoc_fn_test1(x: _);
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    fn assoc_fn_test2(x: _) -> _;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
    fn assoc_fn_test3() -> _;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
}

struct BadStruct<_>(_);
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
trait BadTrait<_> {}
//~^ ERROR expected identifier, found reserved identifier `_`
impl BadTrait<_> for BadStruct<_> {}
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

fn impl_trait() -> impl BadTrait<_> {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
    unimplemented!()
}

struct BadStruct1<_, _>(_);
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR expected identifier, found reserved identifier `_`
//~| ERROR the name `_` is already used
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
struct BadStruct2<_, T>(_, T);
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

type X = Box<_>;
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

struct Struct;
trait Trait<T> {}
impl Trait<usize> for Struct {}
type Y = impl Trait<_>;
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
//~| ERROR the type placeholder `_` is not allowed within types on item signatures
fn foo() -> Y {
    Struct
}
