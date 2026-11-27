#![feature(trait_alias)]

pub trait Hello {
    fn hello(&self);
}

pub struct Hi;

impl Hello for Hi {
    fn hello(&self) {}
}

pub trait Greet = Hello;
