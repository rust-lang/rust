struct Qux<'a> {
    s: &'a String
}

impl<'a> Qux<'a> {
    fn f(&self) {
        self.s.push('x');
        //~^ ERROR cannot borrow borrowed content `*self.s` of immutable binding as mutable
    }
}

fn main() {}
