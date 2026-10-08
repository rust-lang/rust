//@ run-pass
//@ edition:2021..

fn test<'short, 'long>(r: &'short &'long ()) -> impl 'long + Fn() -> &'long () {
    || *r
}

struct Foo(());

impl Foo {
    fn method(&self) {}

    fn assoc(_: &Self) {}
}

fn test2<'a>(foo: &'a Foo) -> impl Fn() -> &'a () {
    || {
        foo.method();
        Foo::assoc(foo);
        &foo.0
    }
}

fn assert_static_fn<Ret>(_: impl Fn() -> Ret + 'static) {}

#[repr(packed)]
struct Packed(u32);

#[allow(unused)]
struct Int(i32);
#[allow(unused)]
struct B<'a>(&'a i32);

struct MyStruct<'a, 'b> {
    a: &'a Int,
    _b: B<'b>,
}

fn foo<'a, 'b, 'c>(m: &'a MyStruct<'b, 'c>) -> impl FnMut() + use<'b> {
    let c = || {
        let _unused = &m.a.0;
    };
    c
}

fn bar<'a, 'b, 'c>(m: &'a MyStruct<'b, 'c>) -> impl FnMut() + use<'b> {
    let c = || {
        let _unused = m.a;
    };
    c
}

fn main() {
    test(&&())();
    test2(&Foo(()))();

    let p = Packed(42);
    assert_static_fn::<u32>(|| p.0);

    foo(&MyStruct { a: &Int(42), _b: B(&42) })();
    bar(&MyStruct { a: &Int(42), _b: B(&42) })();
}
