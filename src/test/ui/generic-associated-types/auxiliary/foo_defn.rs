#![feature(generic_associated_types)]

use std::{future::Future, pin::Pin};

pub trait Foo {
    type Bar: AsRef<()>;
    fn foo(&self) -> Pin<Box<dyn Future<Output = Self::Bar> + '_>>;
}
