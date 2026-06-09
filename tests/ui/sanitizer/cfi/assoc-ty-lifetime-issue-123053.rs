// Regression test for issue 123053, where associated types with lifetimes caused generation of the
// trait object type to fail, causing an ICE.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Zsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer
//@ edition: 2021
//@ no-prefer-dynamic
//@ only-x86_64-unknown-linux-gnu
//@ ignore-backends: gcc
//@ build-pass

trait Iterable {
    type Item<'a>
    where
        Self: 'a;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>
    where
        Self: 'a;

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
}

impl<T> Iterable for [T] {
    type Item<'a> = <std::slice::Iter<'a, T> as Iterator>::Item where T: 'a;
    type Iter<'a> = std::slice::Iter<'a, T> where T: 'a;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.iter()
    }
}

fn get_first<'a, I: Iterable + ?Sized>(it: &'a I) -> Option<I::Item<'a>> {
    it.iter().next()
}

fn main() {
    let v = vec![1, 2, 3];

    assert_eq!(Some(&1), get_first(&*v));
}
