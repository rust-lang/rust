// Regression test for https://github.com/rust-lang/rust/issues/143789
#[must_use::skip]
//~^ ERROR failed to resolve: use of unresolved module or unlinked crate `must_use`
fn main() { }

// Regression test for https://github.com/rust-lang/rust/issues/137590
struct S(#[stable::skip] u8, u16, u32);
//~^ ERROR failed to resolve: use of unresolved module or unlinked crate `stable`
