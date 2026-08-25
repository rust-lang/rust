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

trait Bar1: Foo {
    fn b(&self) -> i32 {
        20
    }

    fn w(&self) -> i32 {
        21
    }
}

trait Bar2: Foo {
    fn c(&self) -> i32 {
        30
    }

    fn v(&self) -> i32 {
        31
    }
}

trait Baz: Bar1 + Bar2 {
    fn d(&self) -> i32 {
        40
    }
}

impl Foo for i32 {
    fn a(&self) -> i32 {
        100
    }
}

impl Bar1 for i32 {
    fn b(&self) -> i32 {
        200
    }
}

impl Bar2 for i32 {
    fn c(&self) -> i32 {
        300
    }
}

impl Baz for i32 {
    fn d(&self) -> i32 {
        400
    }
}

fn main() {
    let baz: &dyn Baz = &1;
    let _: &dyn std::fmt::Debug = baz;
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.d(), 400);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);
    assert_eq!(baz.v(), 31);

    let bar1: &dyn Bar1 = baz;
    let _: &dyn std::fmt::Debug = bar1;
    assert_eq!(*bar1, 1);
    assert_eq!(bar1.a(), 100);
    assert_eq!(bar1.b(), 200);
    assert_eq!(bar1.z(), 11);
    assert_eq!(bar1.y(), 12);
    assert_eq!(bar1.w(), 21);

    let bar2: &dyn Bar2 = baz;
    let _: &dyn std::fmt::Debug = bar2;
    assert_eq!(*bar2, 1);
    assert_eq!(bar2.a(), 100);
    assert_eq!(bar2.c(), 300);
    assert_eq!(bar2.z(), 11);
    assert_eq!(bar2.y(), 12);
    assert_eq!(bar2.v(), 31);

    let foo: &dyn Foo = baz;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);

    let foo: &dyn Foo = bar1;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);

    let foo: &dyn Foo = bar2;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
}
