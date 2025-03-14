#![feature(type_alias_impl_trait)]

pub type WithLifetime<'a> = impl Equals<SelfType = ()>;
#[define_opaque(WithLifetime)]
fn _defining_use<'a>() -> WithLifetime<'a> {}

trait Convert<'a> {
    type Witness;
    fn convert<'b, T: ?Sized>(_proof: &'b Self::Witness, x: &'a T) -> &'b T;
}

impl<'a> Convert<'a> for () {
    type Witness = WithLifetime<'a>;

    fn convert<'b, T: ?Sized>(_proof: &'b WithLifetime<'a>, x: &'a T) -> &'b T {
        // compiler used to think it gets to assume 'a: 'b here because
        // of the `&'b WithLifetime<'a>` argument
        x
        //~^ ERROR lifetime may not live long enough
    }
}

fn extend_lifetime<'a, 'b, T: ?Sized>(x: &'a T) -> &'b T {
    WithLifetime::<'a>::convert_helper::<(), T>(&(), x)
}

trait Equals {
    type SelfType;
    fn convert_helper<'a, 'b, W: Convert<'a, Witness = Self>, T: ?Sized>(
        proof: &'b Self::SelfType,
        x: &'a T,
    ) -> &'b T;
}

impl<S> Equals for S {
    type SelfType = Self;
    fn convert_helper<'a, 'b, W: Convert<'a, Witness = Self>, T: ?Sized>(
        proof: &'b Self,
        x: &'a T,
    ) -> &'b T {
        W::convert(proof, x)
    }
}

fn main() {
    let r;
    {
        let x = String::from("Hello World?");
        r = extend_lifetime(&x);
    }
    println!("{}", r);
}
