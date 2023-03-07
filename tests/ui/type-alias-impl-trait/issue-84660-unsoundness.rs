// Another example from issue #84660, this time weaponized as a safe transmute: an opaque type in an
// impl header being accepted was used to create unsoundness.

#![feature(type_alias_impl_trait)]

trait Foo {}
impl Foo for () {}
type Bar = impl Foo;
fn _defining_use() -> Bar {}

trait Trait<T, In> {
    type Out;
    fn convert(i: In) -> Self::Out;
}

impl<In, Out> Trait<Bar, In> for Out {
    type Out = Out;
    fn convert(_i: In) -> Self::Out {
        unreachable!();
    }
}

impl<In, Out> Trait<(), In> for Out { //~ ERROR conflicting implementations of trait `Trait<Bar, _>`
    type Out = In;
    fn convert(i: In) -> Self::Out {
        i
    }
}

fn transmute<In, Out>(i: In) -> Out {
    <Out as Trait<Bar, In>>::convert(i)
}

fn main() {
    let d;
    {
        let x = "Hello World".to_string();
        d = transmute::<&String, &String>(&x);
    }
    println!("{}", d);
}
