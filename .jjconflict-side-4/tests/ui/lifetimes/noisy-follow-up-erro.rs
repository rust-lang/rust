struct Foo<'c, 'd>(&'c (), &'d ());

impl<'c, 'd> Foo<'c, 'd> {
    fn acc(&mut self, _bar: &Bar) -> &'d () {
        todo!()
    }
}

struct Bar;

impl<'a> Bar {
    fn boom(&self, foo: &mut Foo<'_, '_, 'a>) -> Result<(), &'a ()> {
        //~^ ERROR: struct takes 2 lifetime arguments but 3 lifetime arguments were supplied
        self.bar().map_err(|()| foo.acc(self))?;
        //~^ ERROR: explicit lifetime required in the type of `foo`
        Ok(())
    }
    fn bar(&self) -> Result<(), &'a ()> {
        todo!()
    }
}

fn main() {}
