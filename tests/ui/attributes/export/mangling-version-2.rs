//@ compile-flags: -Zunstable-options -Csymbol-mangling-version=legacy
extern dyn crate libr;
//~^ ERROR `extern dyn` annotation is only allowed with `v0` mangling scheme
//~| ERROR can't find crate for `libr`

fn main() {}
