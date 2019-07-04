// compile-flags:--emit=metadata --error-format=json -Z emit-artifact-notifications
// build-pass (FIXME(62277): could be check-pass?)
// ignore-pass
// ^-- needed because `--pass check` does not emit the output needed.

// A very basic test for the emission of artifact notifications in JSON output.

fn main() {}
