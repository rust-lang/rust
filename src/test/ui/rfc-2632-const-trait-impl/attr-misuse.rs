#![feature(const_trait_impl)]

#[default_method_body_is_const] //~ ERROR attribute should be applied
trait A {
    #[default_method_body_is_const] //~ ERROR attribute should be applied
    fn no_body(self);

    #[default_method_body_is_const]
    fn correct_use(&self) {}
}

#[default_method_body_is_const] //~ ERROR attribute should be applied
fn main() {}
