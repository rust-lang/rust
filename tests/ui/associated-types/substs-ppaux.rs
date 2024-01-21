//
// revisions: verbose normal
//
//[verbose] compile-flags: -Z verbose-internals

trait Foo<'b, 'c, S=u32> {
    fn bar<'a, T>() where T: 'a {}
    fn baz() {}
}

impl<'a,'b,T,S> Foo<'a, 'b, S> for T {}

fn main() {}

fn foo<'z>() where &'z (): Sized {
    let x: () = <i8 as Foo<'static, 'static,  u8>>::bar::<'static, char>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected unit type `()`
    //[verbose]~| found fn item `{fn item <i8 as Foo<ReStatic, ReStatic, u8>>::bar::<ReStatic, char>: fn()}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected unit type `()`
    //[normal]~| found fn item `{fn item <i8 as Foo<'static, 'static, u8>>::bar::<'static, char>: fn()}`


    let x: () = <i8 as Foo<'static, 'static,  u32>>::bar::<'static, char>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected unit type `()`
    //[verbose]~| found fn item `{fn item <i8 as Foo<ReStatic, ReStatic>>::bar::<ReStatic, char>: fn()}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected unit type `()`
    //[normal]~| found fn item `{fn item <i8 as Foo<'static, 'static>>::bar::<'static, char>: fn()}`

    let x: () = <i8 as Foo<'static, 'static,  u8>>::baz;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected unit type `()`
    //[verbose]~| found fn item `{fn item <i8 as Foo<ReStatic, ReStatic, u8>>::baz: fn()}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected unit type `()`
    //[normal]~| found fn item `{fn item <i8 as Foo<'static, 'static, u8>>::baz: fn()}`

    let x: () = foo::<'static>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected unit type `()`
    //[verbose]~| found fn item `{fn item foo::<ReStatic>: fn()}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected unit type `()`
    //[normal]~| found fn item `{fn item foo::<'static>: fn()}`

    <str as Foo<u8>>::bar;
    //[verbose]~^ ERROR the size for values of type
    //[normal]~^^ ERROR the size for values of type
}
