// Test that `#[plugin_registrar]` attribute is gated by `plugin_registrar`
// feature gate.

// The registration function isn't type-checked yet.
#[plugin_registrar]
pub fn registrar() {}
//~^ ERROR compiler plugins are experimental

fn main() {}
