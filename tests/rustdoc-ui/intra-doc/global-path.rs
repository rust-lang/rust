// Doc link path with empty prefix that resolves to "extern prelude" instead of a module.

//@ check-pass
//@ edition:2018

/// [::Unresolved]
//~^ WARN unresolved link to `::Unresolved`
///
/// [::std::unresolved]
//~^ WARN unresolved link to `::std::unresolved`
pub struct Item;
