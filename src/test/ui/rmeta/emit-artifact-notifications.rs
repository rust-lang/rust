// compile-flags:--emit=metadata --error-format=json --json artifacts
// build-pass
// ignore-pass
// ^-- needed because `--pass check` does not emit the output needed.
// ignore-compare-mode-nll

// A very basic test for the emission of artifact notifications in JSON output.

fn main() {}
