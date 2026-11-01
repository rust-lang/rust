// This is a non-regression test for issue #127052 where unreferenced `#[used]` statics in the
// binary crate would be marked as "exported", but not be present in the binary, causing linking
// errors with the MSVC linker.

//@ build-pass: needs linking

#[used]
static FOO: u32 = 0;

fn main() {}
