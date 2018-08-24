// Test that we cannot create objects from unsized types.

trait Foo { fn foo(&self) {} }
impl Foo for str {}
impl Foo for [u8] {}

fn test1<T: ?Sized + Foo>(t: &T) {
    let u: &Foo = t;
    //~^ ERROR the size for values of type
}

fn test2<T: ?Sized + Foo>(t: &T) {
    let v: &Foo = t as &Foo;
    //~^ ERROR the size for values of type
}

fn test3() {
    let _: &[&Foo] = &["hi"];
    //~^ ERROR the size for values of type
}

fn test4(x: &[u8]) {
    let _: &Foo = x as &Foo;
    //~^ ERROR the size for values of type
}

fn main() { }
