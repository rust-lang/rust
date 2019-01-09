struct Foo {
    x: isize,
}

fn main() {
    match Foo { //~ ERROR expected value, found struct `Foo`
        x: 3    //~ ERROR expected one of `=>`, `@`, `if`, or `|`, found `:`
    } {
        Foo { //~ ERROR mismatched types
            x: x //~ ERROR cannot find value `x` in this scope
        } => {} //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `=>`
    }
}
