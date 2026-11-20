#![deny(dead_code)]

struct Used;
struct Unused; //~ ERROR struct `Unused` is never constructed

pub trait PubTrait {
    fn foo(&self) -> Self;
}

impl PubTrait for Used {
    fn foo(&self) -> Self { Used }
}

impl PubTrait for Unused {
    fn foo(&self) -> Self { Unused }
}

trait PriTrait {
    fn foo(&self) -> Self;
}

impl PriTrait for Used {
    fn foo(&self) -> Self { Used }
}

impl PriTrait for Unused {
    fn foo(&self) -> Self { Unused }
}

fn main() {
    let t = Used;
    let _t = <Used as PubTrait>::foo(&t);
    let _t = <Used as PriTrait>::foo(&t);
}
