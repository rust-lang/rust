// Issue 52985: user code provides no use case that allows a type alias `impl Trait`
// We now emit a 'could not find defining uses' error

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait, impl_trait_in_bindings))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type Foo = impl Copy; //~ could not find defining uses

// make compiler happy about using 'Foo'
fn bar(x: Foo) -> Foo { x }

fn main() {
    let _: Foo = std::mem::transmute(0u8); //[min_tait]~ ERROR not permitted here
}
