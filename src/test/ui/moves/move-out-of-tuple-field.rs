#![feature(box_syntax)]

struct Foo(Box<isize>);

fn main() {
    let x: (Box<_>,) = (box 1,);
    let y = x.0;
    let z = x.0; //~ ERROR use of moved value: `x.0`

    let x = Foo(box 1);
    let y = x.0;
    let z = x.0; //~ ERROR use of moved value: `x.0`
}
