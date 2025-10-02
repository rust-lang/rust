// Test that we cannot create objects from unsized types.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Foo { fn foo(&self) {} }
impl Foo for str {}
impl Foo for [u8] {}

fn test1<T: ?Sized + Foo>(t: &T) {
    let u: &dyn Foo = t;
    //~^ ERROR the size for values of type
}

fn test2<T: ?Sized + Foo>(t: &T) {
    let v: &dyn Foo = t as &dyn Foo;
    //~^ ERROR the size for values of type
}

fn test3() {
    let _: &[&dyn Foo] = &["hi"];
    //~^ ERROR the size for values of type
}

fn test4(x: &[u8]) {
    let _: &dyn Foo = x as &dyn Foo;
    //~^ ERROR the size for values of type
}

fn main() { }
