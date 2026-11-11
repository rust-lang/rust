struct Foo<'a>(&'a ());

impl<'a> Foo<'a> {
    fn spam(&mut self, baz: &mut Vec<u32>) {
        match 15 {
            ref Self => (),
            //~^ ERROR expected identifier, found keyword `Self`
        }
    }
}

fn main() {}
