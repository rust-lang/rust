#![crate_type = "lib"]

#[repr(align(16))] //~ ERROR `repr(align)` attributes on functions are unstable
fn requires_alignment() {}
