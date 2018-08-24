// Test that `#![plugin(...)]` attribute is gated by `plugin` feature gate

#![plugin(foo)]
//~^ ERROR compiler plugins are experimental and possibly buggy

fn main() {}
