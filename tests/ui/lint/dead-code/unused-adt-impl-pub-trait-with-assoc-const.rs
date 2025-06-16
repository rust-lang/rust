#![deny(dead_code)]

struct T1; //~ ERROR struct `T1` is never constructed
pub struct T2(i32); //~ ERROR field `0` is never read
struct T3; //~ ERROR struct `T3` is never constructed

trait Trait1 { //~ ERROR trait `Trait1` is never used
    const UNUSED: i32;
    fn unused(&self) {}
    fn construct_self() -> Self;
}

pub trait Trait2 {
    const USED: i32;
    fn used(&self) {}
}

pub trait Trait3 {
    const USED: i32;
    fn construct_self() -> Self;
}

impl Trait1 for T1 {
    const UNUSED: i32 = 0;
    fn construct_self() -> Self {
        Self
    }
}

impl Trait1 for T2 {
    const UNUSED: i32 = 0;
    fn construct_self() -> Self {
        T2(0)
    }
}

impl Trait2 for T1 {
    const USED: i32 = 0;
}

impl Trait2 for T2 {
    const USED: i32 = 0;
}

impl Trait3 for T3 {
    const USED: i32 = 0;
    fn construct_self() -> Self {
        Self
    }
}

fn main() {}
