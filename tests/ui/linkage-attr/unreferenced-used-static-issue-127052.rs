// This is a non-regression test for issue #127052 where unreferenced `#[used]` statics couldn't be
// removed by the MSVC linker, causing linking errors.

//@ build-pass: needs linking
//@ only-msvc

#[used]
static FOO: u32 = 0;
fn main() {}
