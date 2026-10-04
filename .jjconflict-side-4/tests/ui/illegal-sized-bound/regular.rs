struct MutType;

pub trait MutTrait {
    fn function(&mut self)
    where
        Self: Sized;
    //~^ NOTE this has a `Sized` requirement
}

impl MutTrait for MutType {
    fn function(&mut self) {}
}

struct Type;

pub trait Trait {
    fn function(&self)
    where
        Self: Sized;
    //~^ NOTE this has a `Sized` requirement
}

impl Trait for Type {
    fn function(&self) {}
}

fn main() {
    (&mut MutType as &mut dyn MutTrait).function();
    //~^ ERROR the `function` method cannot be invoked on a trait object
    (&Type as &dyn Trait).function();
    //~^ ERROR the `function` method cannot be invoked on a trait object
}
