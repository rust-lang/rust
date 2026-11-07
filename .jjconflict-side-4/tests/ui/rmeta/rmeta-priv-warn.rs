//@ compile-flags: --emit=metadata
//@ no-prefer-dynamic
//@ build-pass (FIXME(62277): could be check-pass?)

#[deny(warnings)]

// Test that we don't get warnings for non-pub main when only emitting metadata.
// (#38273)

fn main() {
}
