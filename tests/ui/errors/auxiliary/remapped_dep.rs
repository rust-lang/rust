//@ compile-flags: --remap-path-prefix={{src-base}}/errors/auxiliary=remapped-aux
// Manually remap, so the remapped path remains in .stderr file.

pub struct SomeStruct {} // This line should be show as part of the error.
