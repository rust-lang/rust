#![feature(generic_associated_types)]

// check-fail

trait Iterable {
    type Item<'x>;
    //~^ Missing bound
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
    //~^ Missing bound
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
    //~^ Missing bound
    fn deserialize2<'a, 'b: 'a>(&self, input: &'a T, input2: &'b T) -> Self::Out<'a>;
}

trait Deserializer3<T, U> {
    type Out<'x, 'y>;
    //~^ Missing bound
    //~^^ Missing bound
    fn deserialize2<'a, 'b>(&self, input: &'a T, input2: &'b U) -> Self::Out<'a, 'b>;
}

trait Deserializer4 {
    type Out<'x>;
    //~^ Missing bound
    fn deserialize<'a, T>(&self, input: &'a T) -> Self::Out<'a>;
}

struct Wrap<T>(T);

trait Des {
    type Out<'x, D>;
    //~^ Missing bound
    fn des<'z, T>(&self, data: &'z Wrap<T>) -> Self::Out<'z, Wrap<T>>;
}
/*
impl Des for () {
    type Out<'x, D> = &'x D;
    fn des<'a, T>(&self, data: &'a Wrap<T>) -> Self::Out<'a, Wrap<T>> {
        data
    }
}
*/

trait Des2 {
    type Out<'x, D>;
    //~^ Missing bound
    fn des<'z, T>(&self, data: &'z Wrap<T>) -> Self::Out<'z, T>;
}
/*
impl Des2 for () {
    type Out<'x, D> = &'x D;
    fn des<'a, T>(&self, data: &'a Wrap<T>) -> Self::Out<'a, T> {
        data
    }
}
*/

trait Des3 {
    type Out<'x, D>;
    //~^ Missing bound
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
