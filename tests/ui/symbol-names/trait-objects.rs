// Ensure that trait objects don't include more than one binder. See #83611

//@ build-fail
//@ revisions: v0
//@[v0]compile-flags: -C symbol-mangling-version=v0
//@[v0]normalize-stderr: "core\[.*?\]" -> "core[HASH]"

#![feature(rustc_attrs)]

trait Bar {
    fn method(&self) {}
}

impl Bar for &dyn FnMut(&u8) {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

trait Foo {
    fn method(&self) {}
}

impl Foo for &(dyn FnMut(&u8) + for<'b> Send) {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

trait Baz {
    fn method(&self) {}
}

impl Baz for &(dyn for<'b> Send + FnMut(&u8)) {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

fn main() {
}
