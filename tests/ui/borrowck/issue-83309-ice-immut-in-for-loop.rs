// rust-lang/rust#83309: The compiler tries to suggest potential
// methods that return `&mut` items. However, when it doesn't
// find such methods, it still tries to add suggestions
// which then fails an assertion later because there was
// no suggestions to make.


fn main() {
    for v in Query.iter_mut() {
        //~^ NOTE this iterator yields `&` references
        *v -= 1;
        //~^ ERROR cannot assign to `*v`, which is behind a `&` reference
        //~| NOTE `v` is a `&` reference, so the data it refers to cannot be written
    }
}

pub struct Query;
pub struct QueryIter<'a>(&'a i32);

impl Query {
    pub fn iter_mut<'a>(&'a mut self) -> QueryIter<'a> {
        todo!();
    }
}

impl<'a> Iterator for QueryIter<'a> {
    type Item = &'a i32;

    fn next(&mut self) -> Option<Self::Item> {
        todo!();
    }
}
