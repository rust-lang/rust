#![feature(generic_associated_types)]

// check-fail

trait Iterable {
    type Item<'x>;
    fn iter<'a>(&'a self) -> Self::Item<'a>;
}

/*
impl<T> Iterable for T {
    type Item<'a> = &'a T;
    fn iter<'a>(&'a self) -> Self::Item<'a> {
        self
    }
}
*/

trait Deserializer<T> {
    type Out<'x>;
    fn deserialize<'a>(&self, input: &'a T) -> Self::Out<'a>;
}

/*
impl<T> Deserializer<T> for () {
    type Out<'a> = &'a T;
    fn deserialize<'a>(&self, input: &'a T) -> Self::Out<'a> { input }
}
*/

trait Deserializer2<T> {
    type Out<'x>;
    fn deserialize2<'a, 'b: 'a>(&self, input: &'a T, input2: &'b T) -> Self::Out<'a>;
}

trait Deserializer3<T, U> {
    type Out<'x, 'y>;
    fn deserialize2<'a, 'b>(&self, input: &'a T, input2: &'b U) -> Self::Out<'a, 'b>;
}

fn main() {}
