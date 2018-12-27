#![allow(duplicate_macro_exports)]

#[macro_export]
macro_rules! foo { ($i:ident) => {} }

#[macro_export]
macro_rules! foo { () => {} }
