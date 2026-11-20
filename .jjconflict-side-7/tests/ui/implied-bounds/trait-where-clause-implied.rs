//@ check-pass

pub trait Trait<'a, 'b> {
    fn method(self, _: &'static &'static ())
    where
        'a: 'b;
}

impl<'a> Trait<'a, 'static> for () {
    // On first glance, this seems like we have the extra implied bound that
    // `'a: 'static`, but we know this from the trait method where clause.
    fn method(self, _: &'static &'a ()) {}
}

fn main() {}
