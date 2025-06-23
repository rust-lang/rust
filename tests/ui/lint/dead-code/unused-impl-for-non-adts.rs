#![deny(dead_code)]

struct Foo; //~ ERROR struct `Foo` is never constructed

trait Trait { //~ ERROR trait `Trait` is never used
    fn foo(&self);
}

impl Trait for Foo {
    fn foo(&self) {}
}

impl Trait for [Foo] {
    fn foo(&self) {}
}
impl<const N: usize> Trait for [Foo; N] {
    fn foo(&self) {}
}

impl Trait for *const Foo {
    fn foo(&self) {}
}
impl Trait for *mut Foo {
    fn foo(&self) {}
}

impl Trait for &Foo {
    fn foo(&self) {}
}
impl Trait for &&Foo {
    fn foo(&self) {}
}
impl Trait for &mut Foo {
    fn foo(&self) {}
}

impl Trait for [&Foo] {
    fn foo(&self) {}
}
impl Trait for &[Foo] {
    fn foo(&self) {}
}
impl Trait for &*const Foo {
    fn foo(&self) {}
}

pub trait Trait2 {
    fn foo(&self);
}

impl Trait2 for Foo {
    fn foo(&self) {}
}

impl Trait2 for [Foo] {
    fn foo(&self) {}
}
impl<const N: usize> Trait2 for [Foo; N] {
    fn foo(&self) {}
}

impl Trait2 for *const Foo {
    fn foo(&self) {}
}
impl Trait2 for *mut Foo {
    fn foo(&self) {}
}

impl Trait2 for &Foo {
    fn foo(&self) {}
}
impl Trait2 for &&Foo {
    fn foo(&self) {}
}
impl Trait2 for &mut Foo {
    fn foo(&self) {}
}

impl Trait2 for [&Foo] {
    fn foo(&self) {}
}
impl Trait2 for &[Foo] {
    fn foo(&self) {}
}
impl Trait2 for &*const Foo {
    fn foo(&self) {}
}

fn main() {}
