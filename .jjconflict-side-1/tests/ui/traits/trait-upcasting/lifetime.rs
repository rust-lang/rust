//@ run-pass

trait Foo: PartialEq<i32> + std::fmt::Debug + Send + Sync {
    fn a(&self) -> i32 {
        10
    }

    fn z(&self) -> i32 { //~ WARN methods `z` and `y` are never used
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

    fn w(&self) -> i32 { //~ WARN method `w` is never used
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

// Note: upcast lifetime means a shorter lifetime.
fn upcast_baz<'a: 'b, 'b, T>(v: Box<dyn Baz + 'a>, _l: &'b T) -> Box<dyn Baz + 'b> {
    v
}
fn upcast_bar<'a: 'b, 'b, T>(v: Box<dyn Bar + 'a>, _l: &'b T) -> Box<dyn Bar + 'b> {
    v
}
fn upcast_foo<'a: 'b, 'b, T>(v: Box<dyn Foo + 'a>, _l: &'b T) -> Box<dyn Foo + 'b> {
    v
}

fn main() {
    let v = Box::new(1);
    let l = &(); // dummy lifetime (shorter than `baz`)

    let baz: Box<dyn Baz> = v.clone();
    let u = upcast_baz(baz, &l);
    assert_eq!(*u, 1);
    assert_eq!(u.a(), 100);
    assert_eq!(u.b(), 200);
    assert_eq!(u.c(), 300);

    let baz: Box<dyn Baz> = v.clone();
    let bar: Box<dyn Bar> = baz;
    let u = upcast_bar(bar, &l);
    assert_eq!(*u, 1);
    assert_eq!(u.a(), 100);
    assert_eq!(u.b(), 200);

    let baz: Box<dyn Baz> = v.clone();
    let foo: Box<dyn Foo> = baz;
    let u = upcast_foo(foo, &l);
    assert_eq!(*u, 1);
    assert_eq!(u.a(), 100);

    let baz: Box<dyn Baz> = v.clone();
    let bar: Box<dyn Bar> = baz;
    let foo: Box<dyn Foo> = bar;
    let u = upcast_foo(foo, &l);
    assert_eq!(*u, 1);
    assert_eq!(u.a(), 100);
}
