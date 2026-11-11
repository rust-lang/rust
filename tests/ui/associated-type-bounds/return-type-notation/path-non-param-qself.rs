#![feature(return_type_notation)]

trait Trait {
    fn method() -> impl Sized;
}

struct Adt;

fn non_param_qself()
where
    <()>::method(..): Send,
    //~^ ERROR ambiguous associated function
    i32::method(..): Send,
    //~^ ERROR ambiguous associated function
    Adt::method(..): Send,
    //~^ ERROR ambiguous associated function
{
}

fn main() {}
