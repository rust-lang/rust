#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

pub trait Tr {
    #[default_method_body_is_const]
    fn a(&self) {}

    #[default_method_body_is_const]
    fn b(&self) {
        ().a()
        //~^ ERROR the trait bound
        //~| ERROR cannot call
    }
}

impl Tr for () {}

fn main() {}
