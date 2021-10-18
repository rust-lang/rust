#![feature(generic_associated_types)]

// check-fail

// We have a `&'a self`, so we need a `Self: 'a`
trait Iterable {
    type Item<'x>;
    //~^ Missing required bounds
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

// We have a `&'a T`, so we need a `T: 'x`
trait Deserializer<T> {
    type Out<'x>;
    //~^ Missing required bounds
    fn deserialize<'a>(&self, input: &'a T) -> Self::Out<'a>;
}

/*
impl<T> Deserializer<T> for () {
    type Out<'a> = &'a T;
    fn deserialize<'a>(&self, input: &'a T) -> Self::Out<'a> { input }
}
*/

// We have a `&'b T` and a `'b: 'a`, so it is implied that `T: 'a`. Therefore, we need a `T: 'x`
trait Deserializer2<T> {
    type Out<'x>;
    //~^ Missing required bounds
    fn deserialize2<'a, 'b: 'a>(&self, input1: &'b T) -> Self::Out<'a>;
}

// We have a `&'a T` and a `&'b U`, so we need a `T: 'x` and a `U: 'y`
trait Deserializer3<T, U> {
    type Out<'x, 'y>;
    //~^ Missing required bounds
    fn deserialize2<'a, 'b>(&self, input: &'a T, input2: &'b U) -> Self::Out<'a, 'b>;
}

// `T` is a param on the function, so it can't be named by the associated type
trait Deserializer4 {
    type Out<'x>;
    fn deserialize<'a, T>(&self, input: &'a T) -> Self::Out<'a>;
}

struct Wrap<T>(T);

// We pass `Wrap<T>` and we see `&'z Wrap<T>`, so we require `D: 'x`
trait Des {
    type Out<'x, D>;
    //~^ Missing required bounds
    fn des<'z, T>(&self, data: &'z Wrap<T>) -> Self::Out<'z, Wrap<T>>;
}
/*
impl Des for () {
    type Out<'x, D> = &'x D; // Not okay
    fn des<'a, T>(&self, data: &'a Wrap<T>) -> Self::Out<'a, Wrap<T>> {
        data
    }
}
*/

// We have `T` and `'z` as GAT substs. Because of `&'z Wrap<T>`, there is an
// implied bound that `T: 'z`, so we require `D: 'x`
trait Des2 {
    type Out<'x, D>;
    //~^ Missing required bounds
    fn des<'z, T>(&self, data: &'z Wrap<T>) -> Self::Out<'z, T>;
}
/*
impl Des2 for () {
    type Out<'x, D> = &'x D;
    fn des<'a, T>(&self, data: &'a Wrap<T>) -> Self::Out<'a, T> {
        &data.0
    }
}
*/

// We see `&'z T`, so we require `D: 'x`
trait Des3 {
    type Out<'x, D>;
    //~^ Missing required bounds
    fn des<'z, T>(&self, data: &'z T) -> Self::Out<'z, T>;
}
/*
impl Des3 for () {
    type Out<'x, D> = &'x D;
    fn des<'a, T>(&self, data: &'a T) -> Self::Out<'a, T> {
          data
    }
}
*/

fn main() {}
