// Regression test for the invalid suggestion in #85735 (the
// underlying issue #21974 still exists here).

trait Foo {}
impl<'a, 'b, T> Foo for T
where
    T: FnMut(&'a ()),
    //~^ ERROR: type annotations needed
    T: FnMut(&'b ()),
{
}

fn main() {}
