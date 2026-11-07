//@ edition:2021
//@ aux-build:bad_on_unimplemented.rs
//@ reference: attributes.diagnostic.on_unimplemented.syntax

// Do not ICE when encountering a malformed `#[diagnostic::on_unimplemented]` annotation in a
// dependency when incorrectly used (#124651).

extern crate bad_on_unimplemented;

use bad_on_unimplemented::*;

fn missing_attr<T: MissingAttr>(_: T) {}
fn duplicate_attr<T: DuplicateAttr>(_: T) {}
fn not_meta_list<T: NotMetaList>(_: T) {}
fn empty<T: Empty>(_: T) {}
fn wrong_delim<T: WrongDelim>(_: T) {}
fn bad_formatter<T: BadFormatter<()>>(_: T) {}
fn no_implicit_args<T: NoImplicitArgs>(_: T) {}
fn missing_arg<T: MissingArg>(_: T) {}
fn bad_arg<T: BadArg>(_: T) {}

fn main() {
    missing_attr(()); //~ ERROR E0277
    duplicate_attr(()); //~ ERROR E0277
    not_meta_list(()); //~ ERROR E0277
    empty(()); //~ ERROR E0277
    wrong_delim(()); //~ ERROR E0277
    bad_formatter(()); //~ ERROR E0277
    no_implicit_args(()); //~ ERROR E0277
    missing_arg(()); //~ ERROR E0277
    bad_arg(()); //~ ERROR E0277
}
