//@ check-pass

#![feature(checked_type_aliases)]
#![deny(missing_debug_implementations)]

pub struct Local;

pub type Alias = Local;

impl std::fmt::Debug for Alias {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }
}

pub struct Generic<T>(T);

pub type GenericAlias<T> = Generic<T>;

impl<T: std::fmt::Debug> std::fmt::Debug for GenericAlias<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }
}

fn main() {}
