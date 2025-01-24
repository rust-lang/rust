//@ known-bug: #128119

trait Trait {
    reuse to_reuse::foo { self }
}

struct S;

mod to_reuse {
    pub fn foo(&self) -> u32 {}
}

impl Trait  S {
    reuse to_reuse::foo { self }
}
