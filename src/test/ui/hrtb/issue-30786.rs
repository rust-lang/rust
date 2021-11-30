// rust-lang/rust#30786: the use of `for<'b> &'b mut A: Stream<Item=T`
// should act as assertion that item does not borrow from its stream;
// but an earlier buggy rustc allowed `.map(|x: &_| x)` which does
// have such an item.
//
// This tests double-checks that we do not allow such behavior to leak
// through again.

// revisions: migrate nll
//[nll]compile-flags: -Z borrowck=mir

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll
// ignore-compare-mode-polonius

pub trait Stream {
    type Item;
    fn next(self) -> Option<Self::Item>;
}

// Example stream
pub struct Repeat(u64);

impl<'a> Stream for &'a mut Repeat {
    type Item = &'a u64;
    fn next(self) -> Option<Self::Item> {
        Some(&self.0)
    }
}

pub struct Map<S, F> {
    stream: S,
    func: F,
}

impl<'a, A, F, T> Stream for &'a mut Map<A, F>
where
    &'a mut A: Stream,
    F: FnMut(<&'a mut A as Stream>::Item) -> T,
{
    type Item = T;
    fn next(self) -> Option<T> {
        match self.stream.next() {
            Some(item) => Some((self.func)(item)),
            None => None,
        }
    }
}

pub struct Filter<S, F> {
    stream: S,
    func: F,
}

impl<'a, A, F, T> Stream for &'a mut Filter<A, F>
where
    for<'b> &'b mut A: Stream<Item = T>, // <---- BAD
    F: FnMut(&T) -> bool,
{
    type Item = <&'a mut A as Stream>::Item;
    fn next(self) -> Option<Self::Item> {
        while let Some(item) = self.stream.next() {
            if (self.func)(&item) {
                return Some(item);
            }
        }
        None
    }
}

pub trait StreamExt
where
    for<'b> &'b mut Self: Stream,
{
    fn mapx<F>(self, func: F) -> Map<Self, F>
    where
        Self: Sized,
        for<'a> &'a mut Map<Self, F>: Stream,
    {
        Map { func: func, stream: self }
    }

    fn filterx<F>(self, func: F) -> Filter<Self, F>
    where
        Self: Sized,
        for<'a> &'a mut Filter<Self, F>: Stream,
    {
        Filter { func: func, stream: self }
    }

    fn countx(mut self) -> usize
    where
        Self: Sized,
    {
        let mut count = 0;
        while let Some(_) = self.next() {
            count += 1;
        }
        count
    }
}

impl<T> StreamExt for T where for<'a> &'a mut T: Stream {}

fn identity<T>(x: &T) -> &T {
    x
}

fn variant1() {
    let source = Repeat(10);

    // Here, the call to `mapx` returns a type `T` to which `StreamExt`
    // is not applicable, because `for<'b> &'b mut T: Stream`) doesn't hold.
    //
    // More concretely, the type `T` is `Map<Repeat, Closure>`, and
    // the where clause doesn't hold because the signature of the
    // closure gets inferred to a signature like `|&'_ Stream| -> &'_`
    // for some specific `'_`, rather than a more generic
    // signature.
    //
    // Why *exactly* we opt for this signature is a bit unclear to me,
    // we deduce it somehow from a reuqirement that `Map: Stream` I
    // guess.
    let map = source.mapx(|x: &_| x);
    let filter = map.filterx(|x: &_| true);
    //[migrate]~^ ERROR the method
    //[nll]~^^ ERROR the method
}

fn variant2() {
    let source = Repeat(10);

    // Here, we use a function, which is not subject to the vagaries
    // of closure signature inference. In this case, we get the error
    // on `countx` as, I think, the test originally expected.
    let map = source.mapx(identity);
    let filter = map.filterx(|x: &_| true);
    let count = filter.countx();
    //[migrate]~^ ERROR the method
    //[nll]~^^ ERROR the method
}

fn main() {}
