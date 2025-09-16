//! Things relevant to the next trait solver.
#![allow(unused, unreachable_pub)]

pub mod abi;
mod consts;
mod def_id;
pub mod fold;
pub mod fulfill;
mod generic_arg;
pub mod generics;
pub mod infer;
pub mod interner;
mod ir_print;
pub mod mapping;
mod opaques;
pub mod predicate;
pub(crate) mod project;
mod region;
mod solver;
mod ty;
pub mod util;

pub use consts::*;
pub use def_id::*;
pub use generic_arg::*;
pub use interner::*;
pub use opaques::*;
pub use predicate::*;
pub use region::*;
pub use solver::*;
pub use ty::*;

pub type Binder<'db, T> = rustc_type_ir::Binder<DbInterner<'db>, T>;
pub type EarlyBinder<'db, T> = rustc_type_ir::EarlyBinder<DbInterner<'db>, T>;
pub type Canonical<'db, T> = rustc_type_ir::Canonical<DbInterner<'db>, T>;
pub type CanonicalVarValues<'db> = rustc_type_ir::CanonicalVarValues<DbInterner<'db>>;
pub type CanonicalVarKind<'db> = rustc_type_ir::CanonicalVarKind<DbInterner<'db>>;
pub type CanonicalQueryInput<'db, V> = rustc_type_ir::CanonicalQueryInput<DbInterner<'db>, V>;
pub type AliasTy<'db> = rustc_type_ir::AliasTy<DbInterner<'db>>;
pub type PolyFnSig<'db> = Binder<'db, rustc_type_ir::FnSig<DbInterner<'db>>>;
pub type TypingMode<'db> = rustc_type_ir::TypingMode<DbInterner<'db>>;

#[cfg(feature = "in-rust-tree")]
use rustc_data_structure::sorted_map::index_map as indexmap;

pub type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
