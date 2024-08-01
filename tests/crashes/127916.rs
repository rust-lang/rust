//@ known-bug: #127916

trait Trait {
    fn foo(&self) -> u32 { 0 }
}

struct F;
struct S;

mod to_reuse {
    pub fn foo(&self) -> u32 {}
}

impl Trait  S {
    reuse to_reuse::foo { self }
}
