// edition:2018
// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn a(self: Pin<&Foo>, f: &Foo) -> &Foo { f }
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ lifetime may not live long enough

    async fn c(self: Pin<&Self>, f: &Foo, g: &Foo) -> (Pin<&Foo>, &Foo) { (self, f) }
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ lifetime may not live long enough
}

type Alias<T> = Pin<T>;
impl Foo {
    async fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> &() { arg }
    //[base]~^ ERROR E0623
    //[nll]~^^ lifetime may not live long enough
}

fn main() {}
