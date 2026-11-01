struct Bar<'a> {
    s: &'a String,
    // use wonky spaces to ensure we are creating the span correctly
    longer_name:   &   'a     Vec<u8>
}

impl<'a> Bar<'a> {
    fn f(&mut self) {
        self.s.push('x');
        //~^ ERROR cannot borrow `*self.s` as mutable, as it is behind a `&` reference

        self.longer_name.push(13);
        //~^ ERROR cannot borrow `*self.longer_name` as mutable, as it is behind a `&` reference
    }
}

fn main() {}
