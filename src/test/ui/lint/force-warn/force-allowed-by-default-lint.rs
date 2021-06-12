// compile-flags: --force-warns elided_lifetimes_in_paths -Zunstable-options
// check-pass

struct Foo<'a> {
    x: &'a u32,
}

fn foo(x: &Foo) {}
//~^ WARN hidden lifetime parameters in types are deprecated

fn main() {}
