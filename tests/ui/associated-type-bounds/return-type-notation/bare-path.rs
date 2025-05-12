#![feature(return_type_notation)]

trait Tr {
    const CONST: usize;

    fn method() -> impl Sized;
}

fn foo<T: Tr>()
where
    T::method(..): Send,
    <T as Tr>::method(..): Send,
{
    let _ = T::CONST::(..);
    //~^ ERROR return type notation not allowed in this position yet
    let _: T::method(..);
    //~^ ERROR return type notation not allowed in this position yet
}

fn main() {}
