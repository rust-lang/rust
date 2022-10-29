// Make sure we honor region constraints when normalizing type annotations.

// check-fail

#![feature(more_qualified_paths)]

trait Trait {
    type Assoc;
}

impl<T> Trait for T
where
    T: 'static,
{
    type Assoc = MyTy<()>;
}

enum MyTy<T> {
    Unit,
    Tuple(),
    Struct {},
    Dumb(T),
}

impl<T> MyTy<T> {
    fn method<X>() {}
    fn method2<X>(&self) {}
}

type Ty<'a> = <&'a () as Trait>::Assoc;

fn test_local<'a>() {
    let _: Ty<'a> = MyTy::Unit;
    //~^ ERROR lifetime may not live long enough
}

fn test_closure_sig<'a, 'b>() {
    |_: Ty<'a>| {};
    //~^ ERROR lifetime may not live long enough
    || -> Option<Ty<'b>> { None };
    //~^ ERROR lifetime may not live long enough
}

fn test_path<'a, 'b, 'c, 'd>() {
    <Ty<'a>>::method::<Ty<'static>>;
    //~^ ERROR lifetime may not live long enough
    <Ty<'static>>::method::<Ty<'b>>;
    //~^ ERROR lifetime may not live long enough

    MyTy::Unit::<Ty<'c>>;
    //~^ ERROR lifetime may not live long enough
}

fn test_call<'a, 'b, 'c>() {
    <Ty<'a>>::method::<Ty<'static>>();
    //~^ ERROR lifetime may not live long enough
    <Ty<'static>>::method::<Ty<'b>>();
    //~^ ERROR lifetime may not live long enough
}

fn test_variants<'a, 'b, 'c>() {
    <Ty<'a>>::Struct {};
    //~^ ERROR lifetime may not live long enough
    <Ty<'b>>::Tuple();
    //~^ ERROR lifetime may not live long enough
    <Ty<'c>>::Unit;
    //~^ ERROR lifetime may not live long enough
}

fn test_method_call<'a>(x: MyTy<()>) {
    x.method2::<Ty<'a>>();
    //~^ ERROR lifetime may not live long enough
}

fn test_struct_path<'a, 'b, 'c, 'd>() {
    struct Struct<T> { x: Option<T>, }

    trait Project {
        type Struct;
        type Enum;
    }
    impl<T> Project for T {
        type Struct = Struct<()>;
        type Enum = MyTy<()>;
    }

    // Resolves to enum variant
    MyTy::<Ty<'a>>::Struct {}; // without SelfTy
    //~^ ERROR lifetime may not live long enough
    <Ty<'b> as Project>::Enum::Struct {}; // with SelfTy
    //~^ ERROR lifetime may not live long enough

    // Resolves to struct and associated type respectively
    Struct::<Ty<'c>> { x: None, }; // without SelfTy
    //~^ ERROR lifetime may not live long enough
    <Ty<'d> as Project>::Struct { x: None, }; // with SelfTy
    //~^ ERROR lifetime may not live long enough
}

fn test_pattern<'a, 'b, 'c>() {
    use MyTy::*;
    match MyTy::Unit {
        Struct::<Ty<'a>> {..} => {},
        //~^ ERROR lifetime may not live long enough
        Tuple::<Ty<'b>> (..) => {},
        //~^ ERROR lifetime may not live long enough
        Unit::<Ty<'c>> => {},
        //~^ ERROR lifetime may not live long enough
        Dumb(_) => {},
    };
}


fn main() {}
