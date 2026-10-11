#![deny(rustdoc::broken_intra_doc_links)]
#![feature(associated_type_defaults)]

pub trait TraitWithDefault {
    type T = usize;
    fn f() -> Self::T {
        0
    }
}

/// Link to [UsesDefaults::T] and [UsesDefaults::f]
//@ has 'associated_defaults/type.UsesDefaults.html' '//a[@href="type.UsesDefaults.html#associatedtype.T"]' 'UsesDefaults::T'
//@ has 'associated_defaults/type.UsesDefaults.html' '//a[@href="type.UsesDefaults.html#method.f"]' 'UsesDefaults::f'
pub struct UsesDefaults;
impl TraitWithDefault for UsesDefaults {}

/// Link to [OverridesDefaults::T] and [OverridesDefaults::f]
//@ has 'associated_defaults/type.OverridesDefaults.html' '//a[@href="type.OverridesDefaults.html#associatedtype.T"]' 'OverridesDefaults::T'
//@ has 'associated_defaults/type.OverridesDefaults.html' '//a[@href="type.OverridesDefaults.html#method.f"]' 'OverridesDefaults::f'
pub struct OverridesDefaults;
impl TraitWithDefault for OverridesDefaults {
    type T = bool;
    fn f() -> bool {
        true
    }
}
