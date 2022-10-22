//
// revisions: verbose normal
//
//[verbose] compile-flags: -Z verbose

trait Foo<'b, 'c, S=u32> {
    fn bar<'a, T>() where T: 'a {}
    fn baz() {}
}

impl<'a,'b,T,S> Foo<'a, 'b, S> for T {}

fn main() {}

fn foo<'z>() where &'z (): Sized {
    let x: () = <i8 as Foo<'static, 'static,  u8>>::bar::<'static, char>;
    //~^ ERROR mismatched types
    //~| expected unit type `()`
    //~| found fn item `fn() {<i8 as Foo<'static, 'static, u8>>::bar::<'static, char>}`


    let x: () = <i8 as Foo<'static, 'static,  u32>>::bar::<'static, char>;
    //~^ ERROR mismatched types
    //~| expected unit type `()`
    //~| found fn item `fn() {<i8 as Foo<'static, 'static>>::bar::<'static, char>}`

    let x: () = <i8 as Foo<'static, 'static,  u8>>::baz;
    //~^ ERROR mismatched types
    //~| expected unit type `()`
    //~| found fn item `fn() {<i8 as Foo<'static, 'static, u8>>::baz}`

    let x: () = foo::<'static>;
    //~^ ERROR mismatched types
    //~| expected unit type `()`
    //~| found fn item `fn() {foo::<'static>}`

    <str as Foo<u8>>::bar;
    //~^ ERROR the size for values of type
}
