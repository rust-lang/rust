#![feature(type_alias_impl_trait)]

trait Static: 'static {}
impl Static for () {}

type Gal<T> = impl Static;
fn _defining<T>() -> Gal<T> {}

trait Callable<Arg> { type Output; }

/// We can infer `<C as Callable<Arg>>::Output: 'static`,
/// because we know `C: 'static` and `Arg: 'static`,
fn box_str<C, Arg>(s: C::Output) -> Box<dyn AsRef<str> + 'static>
where
    Arg: Static,
    C: ?Sized + Callable<Arg> + 'static,
    C::Output: AsRef<str>,
{
    Box::new(s)
}

fn extend_lifetime(s: &str) -> Box<dyn AsRef<str> + 'static> {
    type MalformedTy = dyn for<'a> Callable<Gal<&'a ()>, Output = &'a str>;
    //~^ ERROR binding for associated type `Output` references lifetime `'a`
    box_str::<MalformedTy, _>(s)
}

fn main() {
    let extended = extend_lifetime(&String::from("hello"));
    println!("{}", extended.as_ref().as_ref());
}
