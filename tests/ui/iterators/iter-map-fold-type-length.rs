//@ run-pass
//! Check that type lengths don't explode with `Map` folds.
//!
//! The normal limit is a million, and this test used to exceed 1.5 million, but
//! now we can survive an even tighter limit. Still seems excessive though...
#![type_length_limit = "1327047"]

// Custom wrapper so Iterator methods aren't specialized.
struct Iter<I>(I);

impl<I> Iterator for Iter<I>
where
    I: Iterator
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

fn main() {
    let c = Iter(0i32..10)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .map(|x| x)
        .count();
    assert_eq!(c, 10);
}
