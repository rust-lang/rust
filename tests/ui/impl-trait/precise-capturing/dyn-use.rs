#![feature(precise_capturing)]

fn dyn() -> &'static dyn use<> { &() }
//~^ ERROR expected one of `!`, `(`, `::`, `<`, `where`, or `{`, found keyword `use`
