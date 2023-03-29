#![feature(type_alias_impl_trait)]

// revisions: attr none

pub type Closure = impl FnOnce();

// FIXME(type_alias_impl_trait): cfg_attr doesn't work for `defines` at all yet.
#[cfg_attr(attr, defines(Closure))]
fn main() {
    || -> Closure { || () };
    //~^ ERROR: cannot register hidden type without a `#[defines(...)]` attribute
}
