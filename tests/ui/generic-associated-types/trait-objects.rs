trait StreamingIterator {
    type Item<'a> where Self: 'a;
    fn size_hint(&self) -> (usize, Option<usize>);
    // Uncommenting makes `StreamingIterator` dyn-incompatible.
//    fn next(&mut self) -> Self::Item<'_>;
}

fn min_size(x: &mut dyn for<'a> StreamingIterator<Item<'a> = &'a i32>) -> usize {
    //~^ the trait `StreamingIterator` cannot be made into an object
    x.size_hint().0
    //~^ the trait `StreamingIterator` cannot be made into an object
    //~| the trait `StreamingIterator` cannot be made into an object
}

fn main() {}
