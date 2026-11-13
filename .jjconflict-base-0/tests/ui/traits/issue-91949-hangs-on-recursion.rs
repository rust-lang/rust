//~ ERROR overflow evaluating the requirement `<std::iter::Empty<()> as Iterator>::Item == ()`
//@ build-fail
//@ compile-flags: -Zinline-mir=no -Zwrite-long-types-to-disk=yes

// Regression test for #91949.
// This hanged *forever* on 1.56, fixed by #90423.

#![recursion_limit = "256"]

struct Wrapped<T>(T);

struct IteratorOfWrapped<T, I: Iterator<Item = T>>(I);

impl<T, I: Iterator<Item = T>> Iterator for IteratorOfWrapped<T, I> {
    type Item = Wrapped<T>;
    fn next(&mut self) -> Option<Wrapped<T>> {
        self.0.next().map(Wrapped)
    }
}

fn recurse<T>(elements: T) -> Vec<char>
//~^ WARN function cannot return without recursing
where
    T: Iterator<Item = ()>,
{
    recurse(IteratorOfWrapped(elements).map(|t| t.0))
}

fn main() {
    recurse(std::iter::empty());
}
