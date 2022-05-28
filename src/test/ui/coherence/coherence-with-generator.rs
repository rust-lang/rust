// Test that encountering closures during coherence does not cause issues.
#![feature(type_alias_impl_trait, generators)]
type OpaqueGenerator = impl Sized;
fn defining_use() -> OpaqueGenerator {
    || {
        for i in 0..10 {
            yield i;
        }
    }
}

struct Wrapper<T>(T);
trait Trait {}
impl Trait for Wrapper<OpaqueGenerator> {}
//~^ ERROR cannot implement trait on type alias impl trait
impl<T: Sync> Trait for Wrapper<T> {}
//~^ ERROR conflicting implementations of trait `Trait` for type `Wrapper<OpaqueGenerator>`

fn main() {}
