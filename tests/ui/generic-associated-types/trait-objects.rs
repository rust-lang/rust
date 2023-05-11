// revisions: base extended

#![cfg_attr(extended, feature(generic_associated_types_extended))]
#![cfg_attr(extended, allow(incomplete_features))]

trait StreamingIterator {
    type Item<'a> where Self: 'a;
    fn size_hint(&self) -> (usize, Option<usize>);
    // Uncommenting makes `StreamingIterator` not object safe
//    fn next(&mut self) -> Self::Item<'_>;
}

fn min_size(x: &mut dyn for<'a> StreamingIterator<Item<'a> = &'a i32>) -> usize {
    //[base]~^ the trait `StreamingIterator` cannot be made into an object
    x.size_hint().0
    //[extended]~^ borrowed data escapes
}

fn main() {}
