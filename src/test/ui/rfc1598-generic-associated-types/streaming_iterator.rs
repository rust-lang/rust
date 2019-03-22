#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

// FIXME(#44265): "lifetime argument not allowed on this type" errors will be addressed in a
// follow-up PR

use std::fmt::Display;

trait StreamingIterator {
    type Item<'a>;
    // Applying the lifetime parameter `'a` to `Self::Item` inside the trait.
    fn next<'a>(&'a self) -> Option<Self::Item<'a>>;
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
}

struct Foo<T: StreamingIterator> {
    // Applying a concrete lifetime to the constructor outside the trait.
    bar: <T as StreamingIterator>::Item<'static>,
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
}

// Users can bound parameters by the type constructed by that trait's associated type constructor
// of a trait using HRTB. Both type equality bounds and trait bounds of this kind are valid:
//FIXME(sunjay): This next line should parse and be valid
//fn foo<T: for<'a> StreamingIterator<Item<'a>=&'a [i32]>>(iter: T) { /* ... */ }
fn foo<T>(iter: T) where T: StreamingIterator, for<'a> T::Item<'a>: Display { /* ... */ }
//~^ ERROR lifetime arguments are not allowed for this type [E0109]

// Full example of enumerate iterator

#[must_use = "iterators are lazy and do nothing unless consumed"]
struct StreamEnumerate<I> {
    iter: I,
    count: usize,
}

impl<I: StreamingIterator> StreamingIterator for StreamEnumerate<I> {
    type Item<'a> = (usize, I::Item<'a>);
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
    fn next<'a>(&'a self) -> Option<Self::Item<'a>> {
        //~^ ERROR lifetime arguments are not allowed for this type [E0109]
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

impl<I> StreamEnumerate<I> {
    pub fn new(iter: I) -> Self {
        StreamEnumerate {
            count: 0,
            iter: iter,
        }
    }
}

fn test_stream_enumerate() {
    let v = vec!["a", "b", "c"];
    let se = StreamEnumerate::new(v.iter());
    let a: &str = se.next().unwrap().1;
    for (i, s) in se {
        println!("{} {}", i, s);
    }
    println!("{}", a);
}


fn main() {}
