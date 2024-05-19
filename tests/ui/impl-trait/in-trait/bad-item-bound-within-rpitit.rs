// issue: 114145


pub trait Iterable {
    type Item<'a>
    where
        Self: 'a;

    fn iter(&self) -> impl '_ + Iterator<Item = Self::Item<'_>>;
}

impl<'a, I: 'a + Iterable> Iterable for &'a I {
    type Item<'b> = I::Item<'a>
    where
        'b: 'a;
    //~^ ERROR impl has stricter requirements than trait

    fn iter(&self) -> impl 'a + Iterator<Item = I::Item<'a>> {
        //~^ WARN impl trait in impl method signature does not match trait method signature
        (*self).iter()
    }
}

fn main() {}
