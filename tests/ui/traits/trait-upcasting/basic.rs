//@ run-pass

trait Foo: PartialEq<i32> + std::fmt::Debug + Send + Sync {
    fn a(&self) -> i32 {
        10
    }

    fn z(&self) -> i32 {
        11
    }

    fn y(&self) -> i32 {
        12
    }
}

trait Bar: Foo {
    fn b(&self) -> i32 {
        20
    }

    fn w(&self) -> i32 {
        21
    }
}

trait Baz: Bar {
    fn c(&self) -> i32 {
        30
    }
}

impl Foo for i32 {
    fn a(&self) -> i32 {
        100
    }
}

impl Bar for i32 {
    fn b(&self) -> i32 {
        200
    }
}

impl Baz for i32 {
    fn c(&self) -> i32 {
        300
    }
}

fn main() {
    let baz: &dyn Baz = &1;
    let _: &dyn std::fmt::Debug = baz;
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);

    let bar: &dyn Bar = baz;
    let _: &dyn std::fmt::Debug = bar;
    assert_eq!(*bar, 1);
    assert_eq!(bar.a(), 100);
    assert_eq!(bar.b(), 200);
    assert_eq!(bar.z(), 11);
    assert_eq!(bar.y(), 12);
    assert_eq!(bar.w(), 21);

    let foo: &dyn Foo = baz;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);

    let foo: &dyn Foo = bar;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
}
