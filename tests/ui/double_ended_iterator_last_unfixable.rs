//@no-rustfix
#![warn(clippy::double_ended_iterator_last)]

fn main() {
    let mut index = [true, true, false, false, false, true].iter();
    let subindex = (index.by_ref().take(3), 42);
    let _ = subindex.0.last(); //~ ERROR: called `Iterator::last` on a `DoubleEndedIterator`
}
