#![stable(feature = "a", since = "b")]
//~^ ERROR stability attributes may not be used outside of the standard library
mod inner_private_module {
    // UnnameableTypeAlias isn't marked as reachable, so no stability annotation is required here
    pub type UnnameableTypeAlias = u8;
}

#[stable(feature = "a", since = "b")]
//~^ ERROR stability attributes may not be used outside of the standard library
pub fn f() -> inner_private_module::UnnameableTypeAlias {
    0
}

fn main() {}
