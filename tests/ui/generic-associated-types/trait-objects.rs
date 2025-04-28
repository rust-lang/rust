trait StreamingIterator {
    type Item<'a> where Self: 'a;
    fn size_hint(&self) -> (usize, Option<usize>);
    // Uncommenting makes `StreamingIterator` dyn-incompatible.
//    fn next(&mut self) -> Self::Item<'_>;
}

fn min_size(x: &mut dyn for<'a> StreamingIterator<Item<'a> = &'a i32>) -> usize {
    //~^ ERROR the trait `StreamingIterator` is not dyn compatible
    x.size_hint().0
    //~^ ERROR the trait `StreamingIterator` is not dyn compatible
    //~| ERROR the trait `StreamingIterator` is not dyn compatible
}

fn main() {}
