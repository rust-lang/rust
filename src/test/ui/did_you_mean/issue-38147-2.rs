struct Bar<'a> {
    s: &'a String
}

impl<'a> Bar<'a> {
    fn f(&mut self) {
        self.s.push('x');
        //~^ ERROR cannot borrow `*self.s` as mutable, as it is behind a `&` reference
    }
}

fn main() {}
