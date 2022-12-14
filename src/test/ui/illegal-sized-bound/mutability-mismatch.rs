struct MutType;

pub trait MutTrait {
    fn function(&mut self)
    where
        Self: Sized;
    //~^ this has a `Sized` requirement
}

impl MutTrait for MutType {
    fn function(&mut self) {}
}

struct Type;

pub trait Trait {
    fn function(&self)
    where
        Self: Sized;
    //~^ this has a `Sized` requirement
}

impl Trait for Type {
    fn function(&self) {}
}

fn main() {
    (&MutType as &dyn MutTrait).function();
    //~^ ERROR the `function` method cannot be invoked on a trait object
    //~| NOTE you need `&mut dyn MutTrait` instead of `&dyn MutTrait`
    (&mut Type as &mut dyn Trait).function();
    //~^ ERROR the `function` method cannot be invoked on a trait object
    //~| NOTE you need `&dyn Trait` instead of `&mut dyn Trait`
}
