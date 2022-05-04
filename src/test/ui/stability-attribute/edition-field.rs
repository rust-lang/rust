// aux-build:edition-field-std.rs

extern crate edition_field_std;
use edition_field_std::Iterator;

struct Intersperse;

trait Itertools {
    fn intersperse(&self, separator: ()) -> Intersperse {
        unimplemented!()
    }
}

impl<T> Itertools for T
where
    T: Iterator {}

struct MyIterator;
impl Iterator for MyIterator {}

fn main() {
    let it = MyIterator;
    let _intersperse = it.intersperse(());
}
