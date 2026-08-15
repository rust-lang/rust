trait X {}



trait Iter {
    type Item: X;

    fn into_item(self) -> Self::Item;
    fn as_item(&self) -> &Self::Item;
}

fn bad1<T: Iter>(v: T) -> Box<dyn X + 'static>
{
    let item = v.into_item();
    Box::new(item) //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn bad2<T: Iter>(v: T) -> Box<dyn X + 'static>
    where Box<T::Item> : X
{
    let item: Box<_> = Box::new(v.into_item());
    Box::new(item) //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn bad3<'a, T: Iter>(v: T) -> Box<dyn X + 'a>
{
    let item = v.into_item();
    Box::new(item) //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn bad4<'a, T: Iter>(v: T) -> Box<dyn X + 'a>
    where Box<T::Item> : X
{
    let item: Box<_> = Box::new(v.into_item());
    Box::new(item) //~ ERROR associated type `<T as Iter>::Item` may not live long enough
}

fn ok1<'a, T: Iter>(v: T) -> Box<dyn X + 'a>
    where T::Item : 'a
{
    let item = v.into_item();
    Box::new(item) // OK, T::Item : 'a is declared
}

fn ok2<'a, T: Iter>(v: &T, w: &'a T::Item) -> Box<dyn X + 'a>
    where T::Item : Clone
{
    let item = Clone::clone(w);
    Box::new(item) // OK, T::Item : 'a is implied
}

fn ok3<'a, T: Iter>(v: &'a T) -> Box<dyn X + 'a>
    where T::Item : Clone + 'a
{
    let item = Clone::clone(v.as_item());
    Box::new(item) // OK, T::Item : 'a was declared
}

fn meh1<'a, T: Iter>(v: &'a T) -> Box<dyn X + 'a>
    where T::Item : Clone
{
    // This case is kind of interesting. It's the same as `ok3` but
    // without the explicit declaration. This is valid because `T: 'a
    // => T::Item: 'a`, and the former we can deduce from our argument
    // of type `&'a T`.

    let item = Clone::clone(v.as_item());
    Box::new(item)
}

fn main() {}
