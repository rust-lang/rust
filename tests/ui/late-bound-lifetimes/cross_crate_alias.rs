//@ aux-build:upstream_alias.rs
//@ check-pass

extern crate upstream_alias;

fn foo<'a, T: for<'b> upstream_alias::Trait<'b>>(_: upstream_alias::Alias<'a, T>) -> &'a () {
    todo!()
}

fn main() {}
