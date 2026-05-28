// Check that we correctly infer that b and c must be region
// parameterized because they reference a which requires a region.

type A<'a> = &'a isize;
type B<'a> = Box<A<'a>>;

struct C<'a> {
    f: Box<B<'a>>
}

trait SetF<'a> {
    fn set_f_ok(&mut self, b: Box<B<'a>>);
    fn set_f_bad(&mut self, b: Box<B>);
}

impl<'a> SetF<'a> for C<'a> {
    fn set_f_ok(&mut self, b: Box<B<'a>>) {
        self.f = b;
    }

    fn set_f_bad(&mut self, b: Box<B>) {
        self.f = b;
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
