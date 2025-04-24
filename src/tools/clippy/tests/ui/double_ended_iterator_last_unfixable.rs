//@no-rustfix
#![warn(clippy::double_ended_iterator_last)]

// Should not be linted because applying the lint would move the original iterator. This can only be
// linted if the iterator is used thereafter.
fn main() {
    let mut index = [true, true, false, false, false, true].iter();
    let subindex = (index.by_ref().take(3), 42);
    let _ = subindex.0.last();
    let _ = index.next();
}

fn drop_order() {
    struct S(&'static str);
    impl std::ops::Drop for S {
        fn drop(&mut self) {
            println!("Dropping {}", self.0);
        }
    }

    let v = vec![S("one"), S("two"), S("three")];
    let v = (v.into_iter(), 42);
    println!("Last element is {}", v.0.last().unwrap().0);
    //~^ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
    println!("Done");
}
