//! Tests where the `#[sanitize(..)]` attribute can and cannot be used.

#![feature(sanitize)]
#![feature(extern_types)]
#![feature(impl_trait_in_assoc_type)]
#![warn(unused_attributes)]
#![sanitize(address = "off", thread = "on")]

#[sanitize(address = "off", thread = "on")]
mod submod {}

#[sanitize(address = "off")]
static FOO: u32 = 0;

#[sanitize(thread = "off")] //~ ERROR sanitize attribute not allowed here
static BAR: u32 = 0;

#[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
type MyTypeAlias = ();

#[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
trait MyTrait {
    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    const TRAIT_ASSOC_CONST: u32;

    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    type TraitAssocType;

    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    fn trait_method(&self);

    #[sanitize(address = "off", thread = "on")]
    fn trait_method_with_default(&self) {}

    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    fn trait_assoc_fn();
}

#[sanitize(address = "off")]
impl MyTrait for () {
    const TRAIT_ASSOC_CONST: u32 = 0;

    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    type TraitAssocType = Self;

    #[sanitize(address = "off", thread = "on")]
    fn trait_method(&self) {}
    #[sanitize(address = "off", thread = "on")]
    fn trait_method_with_default(&self) {}
    #[sanitize(address = "off", thread = "on")]
    fn trait_assoc_fn() {}
}

trait HasAssocType {
    type T;
    fn constrain_assoc_type() -> Self::T;
}

impl HasAssocType for () {
    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    type T = impl Copy;
    fn constrain_assoc_type() -> Self::T {}
}

#[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
struct MyStruct {
    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    field: u32,
}

#[sanitize(address = "off", thread = "on")]
impl MyStruct {
    #[sanitize(address = "off", thread = "on")]
    fn method(&self) {}
    #[sanitize(address = "off", thread = "on")]
    fn assoc_fn() {}
}

extern "C" {
    #[sanitize(address = "off", thread = "on")] //~ ERROR sanitize attribute not allowed here
    static X: u32;

    #[sanitize(address = "off", thread = "on")] //~ ERROR sanitize attribute not allowed here
    type T;

    #[sanitize(address = "off", thread = "on")] //~ ERROR sanitize attribute not allowed here
    fn foreign_fn();
}

#[sanitize(address = "off", thread = "on")]
fn main() {
    #[sanitize(address = "off", thread = "on")] //~ ERROR sanitize attribute not allowed here
    let _ = ();

    // Currently not allowed on let statements, even if they bind to a closure.
    // It might be nice to support this as a special case someday, but trying
    // to define the precise boundaries of that special case might be tricky.
    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    let _let_closure = || ();

    // In situations where attributes can already be applied to expressions,
    // the sanitize attribute is allowed on closure expressions.
    let _closure_tail_expr = {
        #[sanitize(address = "off", thread = "on")]
        || ()
    };

    match () {
        #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
        () => (),
    }

    #[sanitize(address = "off")] //~ ERROR sanitize attribute not allowed here
    return ();
}
