//@ run-pass

use std::rc::Rc;
use std::sync::Arc;

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

fn test_box() {
    let v = Box::new(1);

    let baz: Box<dyn Baz> = v.clone();
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);

    let baz: Box<dyn Baz> = v.clone();
    let bar: Box<dyn Bar> = baz;
    assert_eq!(*bar, 1);
    assert_eq!(bar.a(), 100);
    assert_eq!(bar.b(), 200);
    assert_eq!(bar.z(), 11);
    assert_eq!(bar.y(), 12);
    assert_eq!(bar.w(), 21);

    let baz: Box<dyn Baz> = v.clone();
    let foo: Box<dyn Foo> = baz;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);

    let baz: Box<dyn Baz> = v.clone();
    let bar: Box<dyn Bar> = baz;
    let foo: Box<dyn Foo> = bar;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
}

fn test_rc() {
    let v = Rc::new(1);

    let baz: Rc<dyn Baz> = v.clone();
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);

    let baz: Rc<dyn Baz> = v.clone();
    let bar: Rc<dyn Bar> = baz;
    assert_eq!(*bar, 1);
    assert_eq!(bar.a(), 100);
    assert_eq!(bar.b(), 200);
    assert_eq!(bar.z(), 11);
    assert_eq!(bar.y(), 12);
    assert_eq!(bar.w(), 21);

    let baz: Rc<dyn Baz> = v.clone();
    let foo: Rc<dyn Foo> = baz;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);

    let baz: Rc<dyn Baz> = v.clone();
    let bar: Rc<dyn Bar> = baz;
    let foo: Rc<dyn Foo> = bar;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
}

fn test_arc() {
    let v = Arc::new(1);

    let baz: Arc<dyn Baz> = v.clone();
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);

    let baz: Arc<dyn Baz> = v.clone();
    let bar: Arc<dyn Bar> = baz;
    assert_eq!(*bar, 1);
    assert_eq!(bar.a(), 100);
    assert_eq!(bar.b(), 200);
    assert_eq!(bar.z(), 11);
    assert_eq!(bar.y(), 12);
    assert_eq!(bar.w(), 21);

    let baz: Arc<dyn Baz> = v.clone();
    let foo: Arc<dyn Foo> = baz;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);

    let baz: Arc<dyn Baz> = v.clone();
    let bar: Arc<dyn Bar> = baz;
    let foo: Arc<dyn Foo> = bar;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
}

fn main() {
    test_box();
    test_rc();
    test_arc();
}
