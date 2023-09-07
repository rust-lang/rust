struct Qux<'a> {
    s: &'a String
}

impl<'a> Qux<'a> {
    fn f(&self) {
        self.s.push('x');
        //~^ ERROR cannot borrow `*self.s` as mutable, as it is behind an `&` reference
    }
}

fn main() {}
