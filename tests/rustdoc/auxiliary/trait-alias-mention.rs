#![feature(trait_alias)]

pub trait SomeAlias = std::fmt::Debug + std::marker::Copy;
