extern dyn crate foo;
//~^ ERROR can't find crate for `foo`
//~| `extern dyn` annotation is only allowed with `v0` mangling scheme

fn main() {}
