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

// Should not be linted because applying the lint would move the original iterator. This can only be
// linted if the iterator is used thereafter.
fn issue_14139() {
    let mut index = [true, true, false, false, false, true].iter();
    let subindex = index.by_ref().take(3);
    let _ = subindex.last();
    let _ = index.next();

    let mut index = [true, true, false, false, false, true].iter();
    let mut subindex = index.by_ref().take(3);
    let _ = subindex.last();
    let _ = index.next();

    let mut index = [true, true, false, false, false, true].iter();
    let mut subindex = index.by_ref().take(3);
    let subindex = &mut subindex;
    let _ = subindex.last();
    let _ = index.next();

    let mut index = [true, true, false, false, false, true].iter();
    let mut subindex = index.by_ref().take(3);
    let subindex = &mut subindex;
    let _ = subindex.last();
    let _ = index.next();

    let mut index = [true, true, false, false, false, true].iter();
    let (subindex, _) = (index.by_ref().take(3), 42);
    let _ = subindex.last();
    let _ = index.next();
}

fn drop_order() {
    struct DropDeIterator(std::vec::IntoIter<S>);
    impl Iterator for DropDeIterator {
        type Item = S;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }
    }
    impl DoubleEndedIterator for DropDeIterator {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.0.next_back()
        }
    }

    struct S(&'static str);
    impl std::ops::Drop for S {
        fn drop(&mut self) {
            println!("Dropping {}", self.0);
        }
    }

    let v = vec![S("one"), S("two"), S("three")];
    let v = DropDeIterator(v.into_iter());
    println!("Last element is {}", v.last().unwrap().0);
    //~^ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
    println!("Done");
}

fn issue_14444() {
    let mut squares = vec![];
    let last_square = [1, 2, 3]
        .into_iter()
        .map(|x| {
            squares.push(x * x);
            Some(x * x)
        })
        .last();
}
