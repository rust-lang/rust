#![feature(diagnostic_on_unknown)]
#![crate_type = "lib"]

pub mod foo {}

#[diagnostic::on_unknown(note = "the name of this item is a single ident: `{This}`")]
use foo::Foo;
//~^ERROR unresolved import `foo::Foo`
//~|NOTE the name of this item is a single ident: `Foo`
//~|NOTE no `Foo` in `foo`

#[diagnostic::on_unknown(note = "the name of this item is a single ident: `{This}`")]
use foo::foo::Foo;
//~^ERROR unresolved import `foo::foo`
//~|NOTE the name of this item is a single ident: `foo`
//~|NOTE could not find `foo` in `foo`

#[diagnostic::on_unknown(note = "the name of this item is two idents: `{This}`")]
use foo::{Bar, Foo};
//~^ERROR unresolved imports `foo::Bar`, `foo::Foo`
//~|NOTE the name of this item is two idents: `Bar, Foo`
//~|NOTE no `Foo` in `foo`
//~|NOTE no `Bar` in `foo`

#[diagnostic::on_unknown(note = "the name of this item is many idents: `{This}`")]
use foo::{
    Foo,
    //~^ERROR unresolved imports `foo::Foo`, `foo::bar`
    //~|NOTE the name of this item is many idents: `Foo, bar`
    //~|NOTE no `Foo` in `foo`
    bar::{Baz, Biz},
    //~^NOTE could not find `bar` in `foo`
};

#[diagnostic::on_unknown(note = "the name of this is: `{This}`")]
pub use doesnt_exist::*;
//~^ERROR unresolved import `doesnt_exist`
//~|NOTE use of unresolved module or unlinked crate `doesnt_exist`
//~|NOTE the name of this is: `doesnt_exist`

#[diagnostic::on_unknown(note = "the name of this item is a single ident: `{This}`")]
use foo::Foo as DoNotUseForThisParam;
//~^ERROR unresolved import `foo::Foo`
//~|NOTE the name of this item is a single ident: `Foo`
//~|NOTE no `Foo` in `foo`
