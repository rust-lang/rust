// Regression test for issue #73159
// Tests thar we don't suggest replacing 'a with 'static'

struct Foo<'a>(&'a [u8]);

impl<'a> Foo<'a> {
    fn make_it(&self) -> impl Iterator<Item = u8> {
        self.0.iter().copied()
        //~^ ERROR: captures lifetime that does not appear in bounds
    }
}

fn main() {}
