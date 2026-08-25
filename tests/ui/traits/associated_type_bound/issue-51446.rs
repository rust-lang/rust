// Regression test for #51446.
//@ check-pass

trait Foo {
    type Item;
    fn get(&self) -> Self::Item;
}

fn blah<T, F>(x: T, f: F) -> B<T::Item, impl Fn(T::Item)>
where
    T: Foo,
    F: Fn(T::Item),
{
    B { x: x.get(), f }
}

pub struct B<T, F>
where
    F: Fn(T),
{
    pub x: T,
    pub f: F,
}

impl Foo for i32 {
    type Item = i32;
    fn get(&self) -> i32 {
        *self
    }
}

fn main() {
    let _ = blah(0, |_| ());
}
