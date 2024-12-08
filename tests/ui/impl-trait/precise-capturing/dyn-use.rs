fn dyn() -> &'static dyn use<> { &() }
//~^ ERROR expected one of `!`, `(`, `::`, `<`, `where`, or `{`, found keyword `use`
