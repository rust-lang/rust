#![deny(broken_intra_doc_links)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub fn ambiguous() {}

pub struct ambiguous {}

#[macro_export]
macro_rules! multi_conflict { () => {} }

#[allow(non_camel_case_types)]
pub struct multi_conflict {}

pub fn multi_conflict() {}

pub mod type_and_value {}

pub const type_and_value: i32 = 0;

pub mod foo {
    pub enum bar {}

    pub fn bar() {}
}

/// [`ambiguous`] is ambiguous. //~ERROR `ambiguous`
///
/// [ambiguous] is ambiguous. //~ERROR ambiguous
///
/// [`multi_conflict`] is a three-way conflict. //~ERROR `multi_conflict`
///
/// Ambiguous [type_and_value]. //~ERROR type_and_value
///
/// Ambiguous non-implied shortcut link [`foo::bar`]. //~ERROR `foo::bar`
pub struct Docs {}

/// [true] //~ ERROR `true` is both a module and a builtin type
/// [primitive@true]
pub mod r#true {}
