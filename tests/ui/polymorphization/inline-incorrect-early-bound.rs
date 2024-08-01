// This test demonstrates an ICE that may occur when we try to resolve the instance
// of a impl that has different generics than the trait it's implementing. This ensures
// we first check that the args are compatible before resolving the body, just like
// we do in projection before substituting a GAT.
//
// When polymorphization is enabled, we check the optimized MIR for unused parameters.
// This will invoke the inliner, leading to this ICE.

//@ compile-flags: -Zpolymorphize=on -Zinline-mir=yes

trait Trait {
    fn foo<'a, K: 'a>(self, _: K);
}

impl Trait for () {
    #[inline]
    fn foo<K>(self, _: K) {
        //~^ ERROR lifetime parameters or bounds on method `foo` do not match the trait declaration
        todo!();
    }
}

pub fn qux<T>() {
    ().foo(());
}

fn main() {}
