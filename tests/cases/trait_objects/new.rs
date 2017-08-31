#![feature(unboxed_closures)]

pub trait Abc { }

pub struct Def;

impl Abc for Def { }

pub fn a(_: &Abc) { }

pub trait A<T> { }

pub type Something = A<()>;
