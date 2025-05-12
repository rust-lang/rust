//@ revisions: verbose normal
//@[verbose] compile-flags: -Z verbose-internals
//@ dont-require-annotations: NOTE

trait Foo<'b, 'c, S = u32> {
    fn bar<'a, T>()
    where
        T: 'a,
    {
    }
    fn baz() {}
}

impl<'a, 'b, T, S> Foo<'a, 'b, S> for T {}

fn main() {}

fn foo<'z>()
where
    &'z (): Sized,
{
    let x: () = <i8 as Foo<'static, 'static, u8>>::bar::<'static, char>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| NOTE expected unit type `()`
    //[verbose]~| NOTE found fn item `fn() {<i8 as Foo<'static, 'static, u8>>::bar::<'static, char>}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| NOTE expected unit type `()`
    //[normal]~| NOTE found fn item `fn() {<i8 as Foo<'static, 'static, u8>>::bar::<'static, char>}`

    let x: () = <i8 as Foo<'static, 'static, u32>>::bar::<'static, char>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| NOTE expected unit type `()`
    //[verbose]~| NOTE found fn item `fn() {<i8 as Foo<'static, 'static>>::bar::<'static, char>}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| NOTE expected unit type `()`
    //[normal]~| NOTE found fn item `fn() {<i8 as Foo<'static, 'static>>::bar::<'static, char>}`

    let x: () = <i8 as Foo<'static, 'static, u8>>::baz;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| NOTE expected unit type `()`
    //[verbose]~| NOTE found fn item `fn() {<i8 as Foo<'static, 'static, u8>>::baz}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| NOTE expected unit type `()`
    //[normal]~| NOTE found fn item `fn() {<i8 as Foo<'static, 'static, u8>>::baz}`

    let x: () = foo::<'static>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| NOTE expected unit type `()`
    //[verbose]~| NOTE found fn item `fn() {foo::<'static>}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| NOTE expected unit type `()`
    //[normal]~| NOTE found fn item `fn() {foo::<'static>}`

    <str as Foo<u8>>::bar;
    //[verbose]~^ ERROR the trait bound `str: Foo<'?0, '?1, u8>` is not satisfied
    //[normal]~^^ ERROR the trait bound `str: Foo<'_, '_, u8>` is not satisfied
}
