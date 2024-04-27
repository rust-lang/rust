#![feature(inherent_associated_types)]

#![allow(incomplete_features)]
#![deny(rustdoc::broken_intra_doc_links)]

// @has inherent_associated_types/index.html

// @has - '//a/@href' 'enum.Simple.html#associatedtype.Type'
//! [`Simple::Type`]

pub enum Simple {}

impl Simple {
    pub type Type = ();
}

////////////////////////////////////////

// @has 'inherent_associated_types/type.Test0.html' '//a/@href' \
//          'struct.Parametrized.html#associatedtype.Proj'
/// [`Parametrized<bool>::Proj`]
pub type Test0 = ();

// FIXME(inherent_associated_types): The intra-doc link below should point to `Proj-1` not `Proj`.
// The current test checks for the buggy behavior for demonstration purposes.
// The same bug happens for inherent associated functions and constants (see #85960, #93398).
//
// Further, at some point we should reject the intra-doc link `Parametrized::Proj`.
// It currently links to `Parametrized<bool>::Proj`.

// @has 'inherent_associated_types/type.Test1.html'
// @has - '//a/@href' 'struct.Parametrized.html#associatedtype.Proj'
// @!has - '//a/@href' 'struct.Parametrized.html#associatedtype.Proj-1'
/// [`Parametrized<i32>::Proj`]
pub type Test1 = ();

pub struct Parametrized<T>(T);

impl Parametrized<bool> {
    pub type Proj = ();
}

impl Parametrized<i32> {
    pub type Proj = String;
}
