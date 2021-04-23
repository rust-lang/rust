// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type Foo = impl Fn() -> Foo;
//~^ ERROR: could not find defining uses

fn crash(x: Foo) -> Foo {
    x
}

fn main() {

}
