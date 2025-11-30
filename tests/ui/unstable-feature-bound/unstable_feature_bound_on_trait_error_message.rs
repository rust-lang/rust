//@ aux-build:unstable_feature_bound_on_trait.rs

extern crate unstable_feature_bound_on_trait as aux;
//~^ ERROR:  use of unstable library feature `foo`
use aux::Foo;
//~^ ERROR:  use of unstable library feature `foo`

struct Bar{}

impl Foo for Bar {
//~^ ERROR:  use of unstable library feature `foo`
//~^^ ERROR:  use of unstable library feature `foo`
}

fn main() {
}
