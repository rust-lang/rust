struct Pass<'a> {
    s: &'a mut String
}

impl<'a> Pass<'a> {
    fn f(&mut self) {
        self.s.push('x');
    }
}

struct Foo<'a> {
    s: &'a mut String
}

impl<'a> Foo<'a> {
    fn f(&self) {
        self.s.push('x'); //~ ERROR cannot borrow `*self.s` as mutable, as it is behind a `&` reference
    }
}

fn main() {}
