// Make sure we honor region constraints when normalizing type annotations.

//@ check-fail

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
    const CONST: () = ();
    fn method<X>() {}
    fn method2<X>(&self) {}
}

trait TraitAssoc {
    const TRAIT_CONST: ();
    fn trait_method<X>(&self);
}
impl<T> TraitAssoc for T {
    const TRAIT_CONST: () = ();
    fn trait_method<X>(&self) {}
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

fn test_path<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h>() {
    <Ty<'a>>::method::<Ty<'static>>;
    //~^ ERROR lifetime may not live long enough
    <Ty<'static>>::method::<Ty<'b>>;
    //~^ ERROR lifetime may not live long enough

    <Ty<'c>>::trait_method::<Ty<'static>>;
    //~^ ERROR lifetime may not live long enough
    <Ty<'static>>::trait_method::<Ty<'d>>;
    //~^ ERROR lifetime may not live long enough

    <Ty<'e>>::CONST;
    //~^ ERROR lifetime may not live long enough
    <Ty<'f>>::TRAIT_CONST;
    //~^ ERROR lifetime may not live long enough

    <Ty<'static>>::method::<Ty<'static>>;
    <Ty<'static>>::trait_method::<Ty<'static>>;
    <Ty<'static>>::CONST;
    <Ty<'static>>::TRAIT_CONST;

    MyTy::Unit::<Ty<'g>>;
    //~^ ERROR lifetime may not live long enough
    MyTy::<Ty<'h>>::Unit;
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

fn test_method_call<'a, 'b>(x: MyTy<()>) {
    x.method2::<Ty<'a>>();
    //~^ ERROR lifetime may not live long enough
    x.trait_method::<Ty<'b>>();
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

fn test_pattern<'a, 'b, 'c, 'd, 'e, 'f>() {
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
    match MyTy::Unit {
        <Ty<'d>>::Struct {..} => {},
        //~^ ERROR lifetime may not live long enough
        <Ty<'e>>::Tuple (..) => {},
        //~^ ERROR lifetime may not live long enough
        <Ty<'f>>::Unit => {},
        //~^ ERROR lifetime may not live long enough
        Dumb(_) => {},
    };
}


fn main() {}
