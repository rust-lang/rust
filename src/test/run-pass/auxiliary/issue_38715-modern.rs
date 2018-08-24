#![allow(duplicate_macro_exports)]

#[macro_export]
macro_rules! foo_modern { ($i:ident) => {} }

#[macro_export]
macro_rules! foo_modern { () => {} }
