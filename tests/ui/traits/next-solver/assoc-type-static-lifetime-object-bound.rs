//! regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/295

//@ compile-flags: -Znext-solver

#![forbid(unsafe_code)]

trait Tr<'a> {
    type A: 'a;
}

fn f<X: ?Sized + Tr<'static>>(a: <X as Tr<'static>>::A) -> Box<dyn std::any::Any> {
    Box::new(a)
}

fn launder<'b>(r: &'b u8) -> &'static u8 {
    *f::<dyn for<'a> Tr<'a, A = &'b u8>>(r).downcast_ref::<&'static u8>().unwrap()
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    let p;
    {
        let x = Box::new(42u8);
        p = launder(&x);
    }
    println!("{}", *p);
}
