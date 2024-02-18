//@ compile-flags: -Z track-diagnostics
//@ error-pattern: created at
//@ rustc-env:RUST_BACKTRACE=0
//@ failure-status: 101

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr-test ".rs:\d+:\d+" -> ".rs:LL:CC"
//@ normalize-stderr-test "note: rustc .+ running on .+" -> "note: rustc $$VERSION running on $$TARGET"

// The test becomes too flaky if we care about exact args. If `-Z ui-testing`
// from compiletest and `-Z track-diagnostics` from `// compile-flags` at the
// top of this file are present, then assume all args are present.
//@ normalize-stderr-test "note: compiler flags: .*-Z ui-testing.*-Z track-diagnostics" -> "note: compiler flags: ... -Z ui-testing ... -Z track-diagnostics"

fn main() {
    break rust
}
