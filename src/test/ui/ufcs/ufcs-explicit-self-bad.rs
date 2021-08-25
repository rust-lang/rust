struct Foo {
    f: isize,
}



impl Foo {
    fn foo(self: isize, x: isize) -> isize {
        //~^ ERROR invalid `self` parameter type
        self.f + x
    }
}

struct Bar<T> {
    f: T,
}

impl<T> Bar<T> {
    fn foo(self: Bar<isize>, x: isize) -> isize {
        //~^ ERROR invalid `self` parameter type
        x
    }
    fn bar(self: &Bar<usize>, x: isize) -> isize {
        //~^ ERROR invalid `self` parameter type
        x
    }
}

trait SomeTrait {
    fn dummy1(&self);
    fn dummy2(&self);
    fn dummy3(&self);
}

impl<'a, T> SomeTrait for &'a Bar<T> {
    fn dummy1(self: &&'a Bar<T>) { }
    fn dummy2(self: &Bar<T>) {} //~ ERROR mismatched `self` parameter type
    //~^ ERROR mismatched `self` parameter type
    fn dummy3(self: &&Bar<T>) {}
    //~^ ERROR mismatched `self` parameter type
    //~| expected reference `&'a Bar<T>`
    //~| found reference `&Bar<T>`
    //~| lifetime mismatch
    //~| ERROR mismatched `self` parameter type
    //~| expected reference `&'a Bar<T>`
    //~| found reference `&Bar<T>`
    //~| lifetime mismatch
}

fn main() {
    let foo = Box::new(Foo {
        f: 1,
    });
    println!("{}", foo.foo(2));
    let bar = Box::new(Bar {
        f: 1,
    });
    println!("{} {}", bar.foo(2), bar.bar(2));
}
