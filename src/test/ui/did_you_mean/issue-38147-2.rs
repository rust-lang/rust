struct Bar<'a> {
    s: &'a String
}

impl<'a> Bar<'a> {
    fn f(&mut self) {
        self.s.push('x');
        //~^ ERROR cannot borrow borrowed content `*self.s` of immutable binding as mutable
    }
}

fn main() {}
