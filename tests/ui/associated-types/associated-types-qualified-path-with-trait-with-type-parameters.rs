//@ check-pass

trait Foo<T> {
    type Bar;
    fn get_bar() -> <Self as Foo<T>>::Bar;
}

fn main() { }
