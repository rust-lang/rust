// compile-flags: --emit=metadata
// no-prefer-dynamic
// compile-pass

#[deny(warnings)]

// Test that we don't get warnings for non-pub main when only emitting metadata.
// (#38273)

fn main() {
}
