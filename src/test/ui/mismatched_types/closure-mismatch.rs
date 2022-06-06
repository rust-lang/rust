// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait Foo {}

impl<T: Fn(&())> Foo for T {}

fn baz<T: Foo>(_: T) {}

fn main() {
    baz(|_| ());
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR implementation of `FnOnce` is not general enough
    //[nll]~| ERROR mismatched types
}
