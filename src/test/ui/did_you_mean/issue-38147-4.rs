struct Foo<'a> {
    s: &'a mut String
}

fn f(x: usize, f: &Foo) {
    f.s.push('x'); //~ ERROR cannot borrow data mutably
}

fn main() {}
