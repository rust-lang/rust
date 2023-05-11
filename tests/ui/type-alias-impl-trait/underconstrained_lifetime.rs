#![feature(type_alias_impl_trait)]

use std::marker::PhantomData;

trait ProofForConversion<'a, 'b> {
    fn convert<T: ?Sized>(_: PhantomData<Self>, r: &'a T) -> &'b T;
}

impl<'a, 'b> ProofForConversion<'a, 'b> for &'b &'a () {
    fn convert<T: ?Sized>(_: PhantomData<Self>, r: &'a T) -> &'b T {
        r
    }
}

type Converter<'a, 'b> = impl ProofForConversion<'a, 'b>;
//~^ ERROR reference has a longer lifetime than the data it references

// Even _defining_use with an explicit `'a: 'b` compiles fine, too.
fn _defining_use<'a, 'b>(x: &'b &'a ()) -> Converter<'a, 'b> {
    x
}

fn extend_lifetime<'a, 'b, T: ?Sized>(x: &'a T) -> &'b T {
    Converter::<'a, 'b>::convert(PhantomData, x)
}

fn main() {
    let d;
    {
        let x = "Hello World".to_string();
        d = extend_lifetime(&x);
    }
    println!("{}", d);
}
