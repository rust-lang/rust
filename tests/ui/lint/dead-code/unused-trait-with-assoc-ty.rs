#![deny(dead_code)]

struct T1; //~ ERROR struct `T1` is never constructed

trait Foo { type Unused; } //~ ERROR trait `Foo` is never used
impl Foo for T1 { type Unused = Self; }

pub trait Bar { type Used; }
impl Bar for T1 { type Used = Self; }

fn main() {}
