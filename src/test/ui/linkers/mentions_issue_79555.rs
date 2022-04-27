// build-pass
// needs-profiler-support
// needs-rust-lld
// compile-flags: -C linker-flavor=gcc:lld -Z unstable-options -Clink-self-contained=y -Cinstrument-coverage

// Test ensuring that a warning referencing lld known issue 79555 is emitted when:
// - we're asking to use lld, via the enriched gcc linker-flavor
// - the CRT object linking is on
// - either coverage or generating a profile is requested

fn main() {}
