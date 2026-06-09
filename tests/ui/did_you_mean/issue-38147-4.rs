struct Foo<'a> {
    s: &'a mut String
}

fn f(x: usize, f: &Foo) {
    f.s.push('x'); //~ ERROR cannot borrow `*f.s` as mutable, as it is behind a `&` reference
}

fn main() {}
