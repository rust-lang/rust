#![feature(box_syntax, rustc_attrs)]

struct Foo { a: isize, b: isize }

fn main() { #![rustc_error] // rust-lang/rust#49855
    let mut x: Box<_> = box Foo { a: 1, b: 2 };
    let (a, b) = (&mut x.a, &mut x.b);
    //~^ ERROR cannot borrow `x` (via `x.b`) as mutable more than once at a time

    let mut foo: Box<_> = box Foo { a: 1, b: 2 };
    let (c, d) = (&mut foo.a, &foo.b);
    //~^ ERROR cannot borrow `foo` (via `foo.b`) as immutable
}
