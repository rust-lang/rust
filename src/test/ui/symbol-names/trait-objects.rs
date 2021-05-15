// Ensure that trait objects don't include more than one binder. See #83611

// build-fail
// revisions: legacy v0
//[legacy]compile-flags: -Z symbol-mangling-version=legacy
    //[v0]compile-flags: -Z symbol-mangling-version=v0

#![feature(rustc_attrs)]

trait Bar {
    fn method(&self) {}
}

impl Bar for &dyn FnMut(&u8) {
    #[rustc_symbol_name]
    //[legacy]~^ ERROR symbol-name
    //[legacy]~| ERROR demangling
    //[legacy]~| ERROR demangling-alt
    //[v0]~^^^^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

trait Foo {
    fn method(&self) {}
}

impl Foo for &(dyn FnMut(&u8) + for<'b> Send) {
    #[rustc_symbol_name]
    //[legacy]~^ ERROR symbol-name
    //[legacy]~| ERROR demangling
    //[legacy]~| ERROR demangling-alt
    //[v0]~^^^^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

trait Baz {
    fn method(&self) {}
}

impl Baz for &(dyn for<'b> Send + FnMut(&u8)) {
    #[rustc_symbol_name]
    //[legacy]~^ ERROR symbol-name
    //[legacy]~| ERROR demangling
    //[legacy]~| ERROR demangling-alt
    //[v0]~^^^^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

fn main() {
}
