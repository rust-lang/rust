// Regression test for <https://github.com/rust-lang/rust/issues/117718>.
// Ensures that the various "doc(hidden)" reexport cases are correctly handled.

pub mod submodule {
    #[doc(hidden)]
    pub struct Hidden {}
}

#[doc(hidden)]
mod x {
    pub struct Y {}
}

// @has "$.index[*].inner[?(@.import.name=='UsedHidden')].import.source" '"submodule::Hidden"'
// @has "$.index[*].inner[?(@.import.name=='UsedHidden')].import.id" "null"
pub use submodule::Hidden as UsedHidden;
// @has "$.index[*].inner[?(@.import.name=='Z')].import.source" '"x::Y"'
// @has "$.index[*].inner[?(@.import.name=='Z')].import.id" 'null'
pub use x::Y as Z;
