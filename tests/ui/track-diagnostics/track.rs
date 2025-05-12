//@ compile-flags: -Z track-diagnostics
//@ error-pattern: created at
//@ rustc-env:RUST_BACKTRACE=0
//@ failure-status: 101

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@ normalize-stderr: "note: rustc .+ running on .+" -> "note: rustc $$VERSION running on $$TARGET"

// The test becomes too flaky if we care about exact args. If `-Z ui-testing`
// from compiletest and `-Z track-diagnostics` from `// compile-flags` at the
// top of this file are present, then assume all args are present.
//@ normalize-stderr: "note: compiler flags: .*-Z ui-testing.*-Z track-diagnostics" -> "note: compiler flags: ... -Z ui-testing ... -Z track-diagnostics"

fn main() {
    break rust
    //~^ ERROR cannot find value `rust` in this scope
    //~| ERROR `break` outside of a loop or labeled block
    //~| ERROR It looks like you're trying to break rust; would you like some ICE?
}
