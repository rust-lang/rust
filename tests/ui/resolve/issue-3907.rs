// aux-build:issue-3907.rs

extern crate issue_3907;

type Foo = dyn issue_3907::Foo;

struct S {
    name: isize
}

impl Foo for S { //~ ERROR expected trait, found type alias `Foo`
    fn bar() { }
}

fn main() {
    let s = S {
        name: 0
    };
    s.bar();
}
