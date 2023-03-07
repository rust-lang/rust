use std::marker::PhantomData;
struct Foo<'a, 'b, T>(PhantomData<(&'a (), &'b (), T)>)
where
    Foo<'short, 'out, T>: Convert<'a, 'b>;
    //~^ ERROR use of undeclared lifetime name
    //~| ERROR use of undeclared lifetime name `'out`

trait Convert<'a, 'b>: Sized {
    fn cast(&'a self) -> &'b Self;
}
impl<'long: 'short, 'short, T> Convert<'long, 'b> for Foo<'short, 'out, T> {
    //~^ ERROR use of undeclared lifetime name
    //~^^ ERROR use of undeclared lifetime name `'out`
    fn cast(&'long self) -> &'short Foo<'short, 'out, T> {
        //~^ ERROR use of undeclared lifetime name
        self
    }
}

fn badboi<'in_, 'out, T>(x: Foo<'in_, 'out, T>, sadness: &'in_ Foo<'short, 'out, T>) -> &'out T {
    //~^ ERROR use of undeclared lifetime name
    sadness.cast() //~ ERROR mismatched types
}

fn main() {}
