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

// Test for https://github.com/rust-lang/rust/issues/117942
struct Foo {
    _: union  {
        #[rustfmt::skip]
    f: String,
    },
    #[rustfmt::skip]
    _: struct {
    g: i32,
    },
}

fn main() {}
