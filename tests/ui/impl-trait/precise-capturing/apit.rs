#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn hello(_: impl use<> Sized) {}
//~^ ERROR `use<...>` precise capturing syntax not allowed on argument-position `impl Trait`

fn main() {}
