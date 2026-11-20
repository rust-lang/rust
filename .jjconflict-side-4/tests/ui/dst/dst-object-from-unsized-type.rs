// Test that we cannot create objects from unsized types.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Foo { fn foo(&self) {} }
impl Foo for str {}
impl Foo for [u8] {}

fn test1<T: ?Sized + Foo>(t: &T) {
    let u: &dyn Foo = t;
    //[current]~^ ERROR the size for values of type
    //[next]~^^ ERROR the trait bound `&T: CoerceUnsized<&dyn Foo>` is not satisfied in `T`
}

fn test2<T: ?Sized + Foo>(t: &T) {
    let v: &dyn Foo = t as &dyn Foo;
    //[current]~^ ERROR the size for values of type
    //[next]~^^ ERROR the trait bound `&T: CoerceUnsized<&dyn Foo>` is not satisfied in `T`
}

fn test3() {
    let _: &[&dyn Foo] = &["hi"];
    //[current]~^ ERROR the size for values of type
    //[next]~^^ ERROR the trait bound `&str: CoerceUnsized<&dyn Foo>` is not satisfied in `str`
}

fn test4(x: &[u8]) {
    let _: &dyn Foo = x as &dyn Foo;
    //[current]~^ ERROR the size for values of type
    //[next]~^^ ERROR the trait bound `&[u8]: CoerceUnsized<&dyn Foo>` is not satisfied in `[u8]`
}

fn main() { }
