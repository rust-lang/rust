//@ run-pass

use std::fmt::Display;

trait StreamingIterator {
    type Item<'a> where Self: 'a;
    // Applying the lifetime parameter `'a` to `Self::Item` inside the trait.
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}

struct Foo<T: StreamingIterator + 'static> {
    // Applying a concrete lifetime to the constructor outside the trait.
    bar: <T as StreamingIterator>::Item<'static>,
}

// Users can bound parameters by the type constructed by that trait's associated type constructor
// of a trait using HRTB. Both type equality bounds and trait bounds of this kind are valid:
fn _bar<T: for<'a> StreamingIterator<Item<'a>=&'a [i32]>>(_iter: T) { /* ... */ }
fn _foo<T>(_iter: T) where T: StreamingIterator, for<'a> T::Item<'a>: Display { /* ... */ }

// Full example of enumerate iterator

#[must_use = "iterators are lazy and do nothing unless consumed"]
struct StreamEnumerate<I> {
    iter: I,
    count: usize,
}

impl<I: StreamingIterator> StreamingIterator for StreamEnumerate<I> {
    type Item<'a> = (usize, I::Item<'a>) where Self: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        match self.iter.next() {
            None => None,
            Some(val) => {
                let r = Some((self.count, val));
                self.count += 1;
                r
            }
        }
    }
}

impl<I: Iterator> StreamingIterator for I {
    type Item<'a> = <I as Iterator>::Item where Self: 'a;
    fn next(&mut self) -> Option<<I as StreamingIterator>::Item<'_>> {
        Iterator::next(self)
    }
}

impl<I> StreamEnumerate<I> {
    pub fn new(iter: I) -> Self {
        StreamEnumerate {
            count: 0,
            iter,
        }
    }
}

fn test_stream_enumerate() {
    let v = vec!["a", "b", "c"];
    let mut se = StreamEnumerate::new(v.iter());
    while let Some(item) = se.next() {
        assert_eq!(v[item.0], *item.1);
    }
    let x = Foo::<std::slice::Iter<'static, u32>> {
        bar: &0u32,
    };
    assert_eq!(*x.bar, 0u32);
}

fn main() {
    test_stream_enumerate();
}
