// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

use std::cell::RefCell;
use std::io::Read;

fn main() {}

fn inner(mut foo: &[u8]) {
    let refcell = RefCell::new(&mut foo);
    //[base]~^ ERROR `foo` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    //[nll]~^^ ERROR `foo` does not live long enough
    let read = &refcell as &RefCell<dyn Read>;
    //[nll]~^ ERROR lifetime may not live long enough

    read_thing(read);
}

fn read_thing(refcell: &RefCell<dyn Read>) {}
