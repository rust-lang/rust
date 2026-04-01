//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
enum Foo {
    A = -12121,
    B = -2,
    C = -1,
    D = 1,
    E = 2,
    F = 12121,
}

#[allow(dead_code)]
#[repr(i64)]
#[derive(Debug, PartialEq)]
enum Bar {
    A = i64::MIN,
    B = -2,
    C = -1,
    D = 1,
    E = 2,
    F = i64::MAX,
}

fn main() {
    let val: Foo = unsafe { std::mem::transmute::<i16, Foo>(-12121) };
    assert_eq!(val, Foo::A);
    let val: Foo = unsafe { std::mem::transmute::<i16, Foo>(-2) };
    assert_eq!(val, Foo::B);
    let val: Foo = unsafe { std::mem::transmute::<i16, Foo>(-1) };
    assert_eq!(val, Foo::C);
    let val: Foo = unsafe { std::mem::transmute::<i16, Foo>(1) };
    assert_eq!(val, Foo::D);
    let val: Foo = unsafe { std::mem::transmute::<i16, Foo>(2) };
    assert_eq!(val, Foo::E);
    let val: Foo = unsafe { std::mem::transmute::<i16, Foo>(12121) };
    assert_eq!(val, Foo::F);

    let val: Bar = unsafe { std::mem::transmute::<i64, Bar>(i64::MIN) };
    assert_eq!(val, Bar::A);
    let val: Bar = unsafe { std::mem::transmute::<i64, Bar>(-2) };
    assert_eq!(val, Bar::B);
    let val: Bar = unsafe { std::mem::transmute::<i64, Bar>(-1) };
    assert_eq!(val, Bar::C);
    let val: Bar = unsafe { std::mem::transmute::<i64, Bar>(1) };
    assert_eq!(val, Bar::D);
    let val: Bar = unsafe { std::mem::transmute::<i64, Bar>(2) };
    assert_eq!(val, Bar::E);
    let val: Bar = unsafe { std::mem::transmute::<i64, Bar>(i64::MAX) };
    assert_eq!(val, Bar::F);
}
