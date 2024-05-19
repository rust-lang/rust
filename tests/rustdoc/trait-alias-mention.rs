//@ aux-build:trait-alias-mention.rs
//@ build-aux-docs

#![crate_name = "foo"]

extern crate trait_alias_mention;

// @has foo/fn.mention_alias_in_bounds.html '//a[@href="../trait_alias_mention/traitalias.SomeAlias.html"]' 'SomeAlias'
pub fn mention_alias_in_bounds<T: trait_alias_mention::SomeAlias>() {
}
