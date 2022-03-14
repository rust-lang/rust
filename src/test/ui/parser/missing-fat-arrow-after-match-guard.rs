#[derive(PartialEq)]
struct Foo {
    x: isize,
}

fn foo() {
    match () {
        () if f == Foo { x: 42 } {}
        //~^ ERROR expected one of `.`, `=>`, `?`, or an operator, found `{`
        _ => {}
    }
}

fn main() {
    let x = 1;
    let y = 2;
    let value = 3;

    match value {
        Some(x) if x == y {
            //~^ ERROR expected one of `!`, `.`, `::`, `=>`, `?`, or an operator, found `{`
            x
        },
        _ => {
            y
        }
    }
}
