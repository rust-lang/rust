// Test that `#![plugin(...)]` attribute is gated by `plugin` feature gate

#![plugin(foo)]
//~^ ERROR compiler plugins are deprecated
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
