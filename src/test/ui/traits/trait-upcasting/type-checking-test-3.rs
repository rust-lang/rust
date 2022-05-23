// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

#![feature(trait_upcasting)]
#![allow(incomplete_features)]

trait Foo<'a>: Bar<'a> {}
trait Bar<'a> {}

fn test_correct(x: &dyn Foo<'static>) {
    let _ = x as &dyn Bar<'static>;
}

fn test_wrong1<'a>(x: &dyn Foo<'static>, y: &'a u32) {
    let _ = x as &dyn Bar<'a>; // Error
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn test_wrong2<'a>(x: &dyn Foo<'a>) {
    let _ = x as &dyn Bar<'static>; // Error
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {}
