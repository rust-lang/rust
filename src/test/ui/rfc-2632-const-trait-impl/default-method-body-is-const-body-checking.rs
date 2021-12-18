#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

trait Tr {}
impl Tr for () {}

const fn foo<T>() where T: ~const Tr {}

pub trait Foo {
    #[default_method_body_is_const]
    fn foo() {
        foo::<()>();
        //~^ ERROR the trait bound `(): Tr` is not satisfied
    }
}

fn main() {}
