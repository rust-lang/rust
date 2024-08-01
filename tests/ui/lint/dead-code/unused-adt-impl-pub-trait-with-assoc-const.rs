#![deny(dead_code)]

struct T1; //~ ERROR struct `T1` is never constructed
struct T2; //~ ERROR struct `T2` is never constructed
pub struct T3(i32); //~ ERROR struct `T3` is never constructed
pub struct T4(i32); //~ ERROR field `0` is never read

trait Trait1 { //~ ERROR trait `Trait1` is never used
    const UNUSED: i32;
    fn unused(&self) {}
    fn construct_self() -> Self;
}

pub trait Trait2 {
    const MAY_USED: i32;
    fn may_used(&self) {}
}

pub trait Trait3 {
    const MAY_USED: i32;
    fn may_used() -> Self;
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
        Self
    }
}

impl Trait2 for T1 {
    const MAY_USED: i32 = 0;
}

impl Trait2 for T2 {
    const MAY_USED: i32 = 0;
}

impl Trait2 for T3 {
    const MAY_USED: i32 = 0;
}

impl Trait3 for T2 {
    const MAY_USED: i32 = 0;
    fn may_used() -> Self {
        Self
    }
}

impl Trait3 for T4 {
    const MAY_USED: i32 = 0;
    fn may_used() -> Self {
        T4(0)
    }
}

fn main() {}
