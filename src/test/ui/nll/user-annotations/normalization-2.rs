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
}

fn test_call<'a, 'b, 'c>() {
    <Ty<'a>>::method::<Ty<'static>>();
    //~^ ERROR lifetime may not live long enough
    <Ty<'static>>::method::<Ty<'b>>();
    //~^ ERROR lifetime may not live long enough
}

fn test_variants<'a, 'b, 'c>() {
    <Ty<'a>>::Struct {}; //TODO
    //~^ ERROR lifetime may not live long enough
    <Ty<'b>>::Tuple();
    //~^ ERROR lifetime may not live long enough
    <Ty<'c>>::Unit;
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
