extern crate instance_provider_a;
extern crate instance_provider_b;

pub fn foo() {
    instance_provider_a::foo();
    instance_provider_b::foo();
}
