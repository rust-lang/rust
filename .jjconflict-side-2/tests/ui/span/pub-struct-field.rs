// Regression test for issue #26083 and #35435
// Test that span for public struct fields start at `pub`

struct Foo {
    bar: u8,
    pub bar: u8, //~ ERROR is already declared
    pub(crate) bar: u8, //~ ERROR is already declared
}

fn main() {}
