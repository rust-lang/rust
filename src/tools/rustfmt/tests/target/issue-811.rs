trait FooTrait<T>: Sized {
    type Bar: BarTrait<T>;
}

trait BarTrait<T>: Sized {
    type Baz;
    fn foo();
}

type Foo<T: FooTrait> = <<T as FooTrait<U>>::Bar as BarTrait<U>>::Baz;
type Bar<T: BarTrait> = <T as BarTrait<U>>::Baz;

fn some_func<T: FooTrait<U>, U>() {
    <<T as FooTrait<U>>::Bar as BarTrait<U>>::foo();
}

fn some_func<T: BarTrait<U>>() {
    <T as BarTrait<U>>::foo();
}
