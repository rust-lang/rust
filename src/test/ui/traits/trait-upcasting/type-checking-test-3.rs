// ignore-compare-mode-nll
#![feature(trait_upcasting)]
#![allow(incomplete_features)]

trait Foo<'a>: Bar<'a> {}
trait Bar<'a> {}

fn test_correct(x: &dyn Foo<'static>) {
    let _ = x as &dyn Bar<'static>;
}

fn test_wrong1<'a>(x: &dyn Foo<'static>, y: &'a u32) {
    let _ = x as &dyn Bar<'a>; // Error
                               //~^ ERROR mismatched types
}

fn test_wrong2<'a>(x: &dyn Foo<'a>) {
    let _ = x as &dyn Bar<'static>; // Error
                                    //~^ ERROR mismatched types
}

fn main() {}
