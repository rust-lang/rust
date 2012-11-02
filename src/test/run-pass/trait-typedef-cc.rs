// xfail-test FIXME: #3907
// aux-build:trait_typedef_cc.rs
extern mod trait_typedef_cc;

type Foo = trait_typedef_cc::Foo;

struct S {
    name: int
}

impl S: Foo {
    fn bar() { }
}

fn main() {
    let s = S {
        name: 0
    };
    s.bar();
}