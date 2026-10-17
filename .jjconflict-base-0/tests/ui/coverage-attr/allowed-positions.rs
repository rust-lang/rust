//! Tests where the `#[coverage(..)]` attribute can and cannot be used.

//@ reference: attributes.coverage.allowed-positions

#![feature(coverage_attribute)]
#![feature(extern_types)]
#![feature(impl_trait_in_assoc_type)]
#![warn(unused_attributes)]
#![coverage(off)]

#[coverage(off)]
mod submod {}

#[coverage(off)] //~ ERROR attribute cannot be used on
type MyTypeAlias = ();

#[coverage(off)] //~ ERROR attribute cannot be used on
trait MyTrait {
    #[coverage(off)] //~ ERROR attribute cannot be used on
    const TRAIT_ASSOC_CONST: u32;

    #[coverage(off)] //~ ERROR attribute cannot be used on
    type TraitAssocType;

    #[coverage(off)] //~ ERROR attribute cannot be used on
    fn trait_method(&self);

    #[coverage(off)]
    fn trait_method_with_default(&self) {}

    #[coverage(off)] //~ ERROR attribute cannot be used on
    fn trait_assoc_fn();
}

#[coverage(off)]
impl MyTrait for () {
    const TRAIT_ASSOC_CONST: u32 = 0;

    #[coverage(off)] //~ ERROR attribute cannot be used on
    type TraitAssocType = Self;

    #[coverage(off)]
    fn trait_method(&self) {}
    #[coverage(off)]
    fn trait_method_with_default(&self) {}
    #[coverage(off)]
    fn trait_assoc_fn() {}
}

trait HasAssocType {
    type T;
    fn constrain_assoc_type() -> Self::T;
}

impl HasAssocType for () {
    #[coverage(off)] //~ ERROR attribute cannot be used on
    type T = impl Copy;
    fn constrain_assoc_type() -> Self::T {}
}

#[coverage(off)] //~ ERROR attribute cannot be used on
struct MyStruct {
    #[coverage(off)] //~ ERROR attribute cannot be used on
    field: u32,
}

#[coverage(off)]
impl MyStruct {
    #[coverage(off)]
    fn method(&self) {}
    #[coverage(off)]
    fn assoc_fn() {}
}

extern "C" {
    #[coverage(off)] //~ ERROR attribute cannot be used on
    static X: u32;

    #[coverage(off)] //~ ERROR attribute cannot be used on
    type T;

    #[coverage(off)] //~ ERROR attribute cannot be used on
    fn foreign_fn();
}

#[coverage(off)]
fn main() {
    #[coverage(off)] //~ ERROR attribute cannot be used on
    let _ = ();

    // Currently not allowed on let statements, even if they bind to a closure.
    // It might be nice to support this as a special case someday, but trying
    // to define the precise boundaries of that special case might be tricky.
    #[coverage(off)] //~ ERROR attribute cannot be used on
    let _let_closure = || ();

    // In situations where attributes can already be applied to expressions,
    // the coverage attribute is allowed on closure expressions.
    let _closure_tail_expr = {
        #[coverage(off)]
        || ()
    };

    // Applying attributes to arbitrary expressions requires an unstable
    // feature, but if that feature were enabled then this would be allowed.
    let _closure_expr = #[coverage(off)] || ();
    //~^ ERROR attributes on expressions are experimental [E0658]

    match () {
        #[coverage(off)] //~ ERROR attribute cannot be used on
        () => (),
    }

    #[coverage(off)] //~ ERROR attribute cannot be used on
    return ();
}
