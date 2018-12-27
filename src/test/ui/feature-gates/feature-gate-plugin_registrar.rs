// Test that `#[plugin_registrar]` attribute is gated by `plugin_registrar`
// feature gate.

// the registration function isn't typechecked yet
#[plugin_registrar]
pub fn registrar() {}
//~^ ERROR compiler plugins are experimental
fn main() {}
