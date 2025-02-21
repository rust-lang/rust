#![warn(clippy::double_ended_iterator_last)]

// Typical case
pub fn last_arg(s: &str) -> Option<&str> {
    s.split(' ').last() //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
}

fn main() {
    // General case
    struct DeIterator;
    impl Iterator for DeIterator {
        type Item = ();
        fn next(&mut self) -> Option<Self::Item> {
            Some(())
        }
    }
    impl DoubleEndedIterator for DeIterator {
        fn next_back(&mut self) -> Option<Self::Item> {
            Some(())
        }
    }
    let _ = DeIterator.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
    // Should not apply to other methods of Iterator
    let _ = DeIterator.count();

    // Should not apply to simple iterators
    struct SimpleIterator;
    impl Iterator for SimpleIterator {
        type Item = ();
        fn next(&mut self) -> Option<Self::Item> {
            Some(())
        }
    }
    let _ = SimpleIterator.last();

    // Should not apply to custom implementations of last()
    struct CustomLast;
    impl Iterator for CustomLast {
        type Item = ();
        fn next(&mut self) -> Option<Self::Item> {
            Some(())
        }
        fn last(self) -> Option<Self::Item> {
            Some(())
        }
    }
    impl DoubleEndedIterator for CustomLast {
        fn next_back(&mut self) -> Option<Self::Item> {
            Some(())
        }
    }
    let _ = CustomLast.last();
}

fn issue_14139() {
    let mut index = [true, true, false, false, false, true].iter();
    let subindex = index.by_ref().take(3);
    let _ = subindex.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`

    let mut index = [true, true, false, false, false, true].iter();
    let mut subindex = index.by_ref().take(3);
    let _ = subindex.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`

    let mut index = [true, true, false, false, false, true].iter();
    let mut subindex = index.by_ref().take(3);
    let subindex = &mut subindex;
    let _ = subindex.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`

    let mut index = [true, true, false, false, false, true].iter();
    let mut subindex = index.by_ref().take(3);
    let subindex = &mut subindex;
    let _ = subindex.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`

    let mut index = [true, true, false, false, false, true].iter();
    let (subindex, _) = (index.by_ref().take(3), 42);
    let _ = subindex.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
}

fn drop_order() {
    struct S(&'static str);
    impl std::ops::Drop for S {
        fn drop(&mut self) {
            println!("Dropping {}", self.0);
        }
    }

    let v = vec![S("one"), S("two"), S("three")];
    let v = v.into_iter();
    println!("Last element is {}", v.last().unwrap().0);
    //~^ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
    println!("Done");
}
