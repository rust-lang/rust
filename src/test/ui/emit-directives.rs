// ignore-tidy-linelength
// compile-flags:--emit=metadata --error-format=json -Z emit-directives
// compile-pass
//
// Normalization is required to eliminated minor path and filename differences
// across platforms.
// normalize-stderr-test: "metadata file written: .*/emit-directives" -> "metadata file written: .../emit-directives"
// normalize-stderr-test: "emit-directives(\.\w*)?/a(\.\w*)?" -> "emit-directives/a"

// A very basic test for the emission of build directives in JSON output.

fn main() {}
