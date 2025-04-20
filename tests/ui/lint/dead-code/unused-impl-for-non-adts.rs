#![deny(dead_code)]

struct Foo; //~ ERROR struct `Foo` is never constructed

trait Trait { //~ ERROR trait `Trait` is never used
    fn foo(&self) {}
}

impl Trait for Foo {}

impl Trait for [Foo] {}
impl<const N: usize> Trait for [Foo; N] {}

impl Trait for *const Foo {}
impl Trait for *mut Foo {}

impl Trait for &Foo {}
impl Trait for &&Foo {}
impl Trait for &mut Foo {}

impl Trait for [&Foo] {}
impl Trait for &[Foo] {}
impl Trait for &*const Foo {}

pub trait Trait2 {
    fn foo(&self) {}
}

impl Trait2 for Foo {}

impl Trait2 for [Foo] {}
impl<const N: usize> Trait2 for [Foo; N] {}

impl Trait2 for *const Foo {}
impl Trait2 for *mut Foo {}

impl Trait2 for &Foo {}
impl Trait2 for &&Foo {}
impl Trait2 for &mut Foo {}

impl Trait2 for [&Foo] {}
impl Trait2 for &[Foo] {}
impl Trait2 for &*const Foo {}

fn main() {}
