// Tests that we don't allow closures to constrain opaque types
// unless their surrounding item has the opaque type in its signature.

#![feature(type_alias_impl_trait)]

pub type Closure = impl FnOnce();

fn main() {
    || -> Closure { || () };
    //~^ ERROR: opaque type constrained without being represented in the signature
}
