//@ check-pass

trait Foo {
    type Assoc;

    fn do_it(_: &Self::Assoc)
    where
        for<'a> Self: Baz<'a>;
}

trait Baz<'a>: Foo {}

impl Foo for () {
    type Assoc = Inherent;

    // Ensure that the `for<'a> Self: Baz<'a>` predicate, which has
    // a supertrait `for<'a> Self: Foo`, does not cause us to fail
    // to normalize `Self::Assoc`.
    fn do_it(x: &Self::Assoc)
    where
        for<'a> Self: Baz<'a>,
    {
        x.inherent();
    }
}

struct Inherent;
impl Inherent {
    fn inherent(&self) {}
}

// This trivial bound doesn't hold, but the unused lifetime tripped up that check after #117589, and
// showed up in its crater results (in `soa-derive 0.13.0`).
fn do_it()
where
    for<'a> Inherent: Clone,
{
}

fn main() {}
