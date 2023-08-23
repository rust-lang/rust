// Test for issue 85480
// Pretty print anonymous struct and union types

// pp-exact
// pretty-compare-only

struct Foo {
    _: union  {
        _: struct  {
            a: u8,
            b: u16,
        },
        c: u32,
    },
    d: u64,
    e: f32,
}

fn main() {}
