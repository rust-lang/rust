// Test for the Path::name field within a single crate.
//
// See https://github.com/rust-lang/rust/issues/135600
// and https://github.com/rust-lang/rust/pull/134880#issuecomment-2596386111
//
//@ aux-build: defines_and_reexports.rs
extern crate defines_and_reexports;

mod priv_mod {
    pub struct InPrivMod;
}

pub mod pub_mod {
    pub struct InPubMod;
}

use priv_mod::InPrivMod as InPrivMod3;
pub use priv_mod::{InPrivMod, InPrivMod as InPrivMod2};
use pub_mod::InPubMod as InPubMod3;
pub use pub_mod::{InPubMod, InPubMod as InPubMod2};

//@ is "$.index[?(@.name=='T0')].inner.type_alias.type.resolved_path.path" '"priv_mod::InPrivMod"'
pub type T0 = priv_mod::InPrivMod;
//@ is "$.index[?(@.name=='T1')].inner.type_alias.type.resolved_path.path" '"InPrivMod"'
pub type T1 = InPrivMod;
//@ is "$.index[?(@.name=='T2')].inner.type_alias.type.resolved_path.path" '"InPrivMod2"'
pub type T2 = InPrivMod2;
//@ is "$.index[?(@.name=='T3')].inner.type_alias.type.resolved_path.path" '"priv_mod::InPrivMod"'
pub type T3 = InPrivMod3;

//@ is "$.index[?(@.name=='U0')].inner.type_alias.type.resolved_path.path" '"pub_mod::InPubMod"'
pub type U0 = pub_mod::InPubMod;
//@ is "$.index[?(@.name=='U1')].inner.type_alias.type.resolved_path.path" '"InPubMod"'
pub type U1 = InPubMod;
//@ is "$.index[?(@.name=='U2')].inner.type_alias.type.resolved_path.path" '"InPubMod2"'
pub type U2 = InPubMod2;
//@ is "$.index[?(@.name=='U3')].inner.type_alias.type.resolved_path.path" '"pub_mod::InPubMod"'
pub type U3 = InPubMod3;

// Check we only have paths for structs at their original path
//@ ismany "$.paths[?(@.crate_id==0 && @.kind=='struct')].path" '["path_name", "priv_mod", "InPrivMod"]' '["path_name", "pub_mod", "InPubMod"]'

pub use defines_and_reexports::{InPrivMod as XPrivMod, InPubMod as XPubMod};
use defines_and_reexports::{InPrivMod as XPrivMod2, InPubMod as XPubMod2};

//@ is "$.index[?(@.name=='X0')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::m1::InPubMod"'
pub type X0 = defines_and_reexports::m1::InPubMod;
//@ is "$.index[?(@.name=='X1')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::InPubMod"'
pub type X1 = defines_and_reexports::InPubMod;
//@ is "$.index[?(@.name=='X2')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::InPubMod2"'
pub type X2 = defines_and_reexports::InPubMod2;
//@ is "$.index[?(@.name=='X3')].inner.type_alias.type.resolved_path.path" '"XPubMod"'
pub type X3 = XPubMod;
// N.B. This isn't the path as used *or* the original path!
//@ is "$.index[?(@.name=='X4')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::InPubMod"'
pub type X4 = XPubMod2;

//@ is "$.index[?(@.name=='Y1')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::InPrivMod"'
pub type Y1 = defines_and_reexports::InPrivMod;
//@ is "$.index[?(@.name=='Y2')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::InPrivMod2"'
pub type Y2 = defines_and_reexports::InPrivMod2;
//@ is "$.index[?(@.name=='Y3')].inner.type_alias.type.resolved_path.path" '"XPrivMod"'
pub type Y3 = XPrivMod;
//@ is "$.index[?(@.name=='Y4')].inner.type_alias.type.resolved_path.path" '"defines_and_reexports::InPrivMod"'
pub type Y4 = XPrivMod2;

// For foreign items, $.paths contains the *origional* path, even if it's not publicly
// assessable. This should probably be changed.

//@ has "$.paths[*].path" '["defines_and_reexports", "m1", "InPubMod"]'
//@ has "$.paths[*].path" '["defines_and_reexports", "m2", "InPrivMod"]'
//@ !has "$.paths[*].path" '["defines_and_reexports", "InPubMod"]'
//@ !has "$.paths[*].path" '["defines_and_reexports", "InPrivMod"]'

// Tests for the example in the docs of Path::name.
// If these change, chage the docs.
//@ is "$.index[?(@.name=='Vec1')].inner.type_alias.type.resolved_path.path" '"std::vec::Vec"'
pub type Vec1 = std::vec::Vec<i32>;
//@ is "$.index[?(@.name=='Vec2')].inner.type_alias.type.resolved_path.path" '"Vec"'
pub type Vec2 = Vec<i32>;
//@ is "$.index[?(@.name=='Vec3')].inner.type_alias.type.resolved_path.path" '"std::prelude::v1::Vec"'
pub type Vec3 = std::prelude::v1::Vec<i32>;
