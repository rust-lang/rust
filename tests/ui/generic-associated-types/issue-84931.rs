//@ check-fail

trait StreamingIter {
    type Item<'a> where Self: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item::<'a>>;
}

struct StreamingSliceIter<'a, T> {
    idx: usize,
    data: &'a mut [T],
}

impl<'b, T: 'b> StreamingIter for StreamingSliceIter<'b, T> {
    type Item<'a> = &'a mut T;
    //~^ ERROR: the parameter type
    //~| ERROR: does not fulfill the required lifetime
    fn next(&mut self) -> Option<&mut T> {
        loop {}
    }
}

fn main() {}
