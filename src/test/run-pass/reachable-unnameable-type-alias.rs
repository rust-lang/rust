#![feature(staged_api)]
#![stable(feature = "a", since = "b")]

mod inner_private_module {
    // UnnameableTypeAlias isn't marked as reachable, so no stability annotation is required here
    pub type UnnameableTypeAlias = u8;
}

#[stable(feature = "a", since = "b")]
pub fn f() -> inner_private_module::UnnameableTypeAlias {
    0
}

fn main() {}
