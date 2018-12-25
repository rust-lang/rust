// compile-flags: -Z parse-only

struct Foo {
    x: isize,
}

fn main() {
    match Foo {
        x: 3    //~ ERROR expected one of `=>`, `@`, `if`, or `|`, found `:`
    } {
        Foo {
            x: x
        } => {} //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `=>`
    }
}
