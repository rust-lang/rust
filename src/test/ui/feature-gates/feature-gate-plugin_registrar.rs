// Test that `#[plugin_registrar]` attribute is gated by `plugin_registrar`
// feature gate.

// the registration function isn't typechecked yet
#[plugin_registrar]
//~^ ERROR compiler plugins are deprecated
//~| WARN use of deprecated attribute `plugin_registrar`: compiler plugins are deprecated
pub fn registrar() {}
//~^ ERROR compiler plugins are experimental and possibly buggy

fn main() {}
