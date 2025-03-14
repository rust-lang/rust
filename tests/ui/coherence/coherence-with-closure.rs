// Test that encountering closures during coherence does not cause issues.
#![feature(type_alias_impl_trait)]
type OpaqueClosure = impl Sized;
#[define_opaque(OpaqueClosure)]
fn defining_use() -> OpaqueClosure {
    || ()
}

struct Wrapper<T>(T);
trait Trait {}
impl Trait for Wrapper<OpaqueClosure> {}
impl<T: Sync> Trait for Wrapper<T> {}
//~^ ERROR conflicting implementations of trait `Trait` for type `Wrapper<OpaqueClosure>`

fn main() {}
