// Regression test for issue #26083
// Test that span for public struct fields start at `pub` instead of the identifier

struct Foo {
    pub bar: u8,

    pub
    //~^ error: field `bar` is already declared [E0124]
    bar: u8,

    pub bar:
    //~^ error: field `bar` is already declared [E0124]
    u8,

    bar:
    //~^ error: field `bar` is already declared [E0124]
    u8,
}

fn main() { }
