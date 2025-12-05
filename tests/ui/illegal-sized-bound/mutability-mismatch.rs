struct MutType;

pub trait MutTrait {
    fn function(&mut self)
    where
        Self: Sized;
}

impl MutTrait for MutType {
    fn function(&mut self) {}
}

struct Type;

pub trait Trait {
    fn function(&self)
    where
        Self: Sized;
}

impl Trait for Type {
    fn function(&self) {}
}

fn main() {
    (&MutType as &dyn MutTrait).function();
    //~^ ERROR the `function` method cannot be invoked on `&dyn MutTrait`
    //~| HELP you need `&mut dyn MutTrait` instead of `&dyn MutTrait`
    (&mut Type as &mut dyn Trait).function();
    //~^ ERROR the `function` method cannot be invoked on `&mut dyn Trait`
    //~| HELP you need `&dyn Trait` instead of `&mut dyn Trait`
}
