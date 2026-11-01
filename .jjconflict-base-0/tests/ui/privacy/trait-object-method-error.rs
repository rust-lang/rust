//! Trait objects only allow access to methods defined in the trait.

trait MyTrait {
    fn trait_method(&mut self);
}

struct ImplType;

impl MyTrait for ImplType {
    fn trait_method(&mut self) {}
}

impl ImplType {
    fn struct_impl_method(&mut self) {}
}

fn main() {
    let obj: Box<dyn MyTrait> = Box::new(ImplType);
    obj.struct_impl_method(); //~ ERROR no method named `struct_impl_method` found
}
