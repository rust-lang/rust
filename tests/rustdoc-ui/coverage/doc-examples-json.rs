//@ check-pass
//@ compile-flags:-Z unstable-options --output-format json --show-coverage

// This check ensures that only one doc example is counted since they're "optional" on
// certain items.

/// ```
/// let x = 12;
/// ```
pub const Foo: u32 = 0;

/// doc
pub const Bar: u32 = 0;
