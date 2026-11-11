struct Foo {
    a: i32,
    b: i32,
    bar: Bar,
}

struct Bar;

impl Bar {
    fn f<F>(&self, mut f: F)
    where F: FnMut()
    {
        f();
    }
}

fn main() {
    let mut foo = Foo { a: 1, b: 2, bar: Bar };
    let a = &mut foo.a;
    let b = &mut foo.b;
    (|| { // works
        *if true {a} else {b} = 42;
    })();

    let mut foo = Foo { a: 1, b: 2, bar: Bar };
    let a = &mut foo.a;
    let b = &mut foo.b;
    foo.bar.f(|| { // doesn't work
        *if true {a} else {b} = 42;
        //~^ ERROR: cannot move out of `a`, a captured variable in an `FnMut` closure
        //~| ERROR: cannot move out of `b`, a captured variable in an `FnMut` closure
    });
}
