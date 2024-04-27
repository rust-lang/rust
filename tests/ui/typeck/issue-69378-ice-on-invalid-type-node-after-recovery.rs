// Regression test for #69378: no type for node after struct parse recovery

struct Foo { 0: u8 } //~ ERROR expected identifier

fn test(f: Foo) {
    Foo{foo: 4, ..f};
}

fn main() {}
