#![deny(unreachable_patterns)]

struct Foo {
    x: isize,
    y: isize,
}

pub fn main() {
    let a = Foo { x: 1, y: 2 };
    match a {
        Foo { x: _x, y: _y } => (),
        Foo { .. } => () //~ ERROR unreachable pattern
    }

}
