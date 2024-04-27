// Issue #5886: a complex instance of issue #2687.

trait Iterator<A> {
    fn next(&mut self) -> Option<A>;
}

trait IteratorUtil<A>: Sized
{
    fn zip<B, U: Iterator<U>>(self, other: U) -> ZipIterator<Self, U>;
}

impl<A, T: Iterator<A>> IteratorUtil<A> for T {
    fn zip<B, U: Iterator<B>>(self, other: U) -> ZipIterator<T, U> {
    //~^ ERROR E0276
        ZipIterator{a: self, b: other}
    }
}

struct ZipIterator<T, U> {
    a: T, b: U
}

fn main() {}
