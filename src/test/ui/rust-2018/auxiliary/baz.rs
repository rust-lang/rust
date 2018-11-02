// This file is used as part of the local-path-suggestions.rs and
// the trait-import-suggestions.rs test.

pub mod foobar {
    pub struct Baz;
}

pub trait BazTrait {
    fn extern_baz(&self) { }
}

impl BazTrait for u32 { }
