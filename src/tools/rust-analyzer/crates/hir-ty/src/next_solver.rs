//! Things relevant to the next trait solver.

pub mod abi;
mod consts;
mod def_id;
pub mod fold;
pub mod fulfill;
mod generic_arg;
pub mod generics;
pub mod infer;
pub(crate) mod inspect;
pub mod interner;
mod ir_print;
pub mod normalize;
pub mod obligation_ctxt;
mod opaques;
pub mod predicate;
mod region;
mod solver;
mod structural_normalize;
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

pub use crate::lower::ImplTraitIdx;
pub use rustc_ast_ir::Mutability;

pub type Binder<'db, T> = rustc_type_ir::Binder<DbInterner<'db>, T>;
pub type EarlyBinder<'db, T> = rustc_type_ir::EarlyBinder<DbInterner<'db>, T>;
pub type Canonical<'db, T> = rustc_type_ir::Canonical<DbInterner<'db>, T>;
pub type CanonicalVarValues<'db> = rustc_type_ir::CanonicalVarValues<DbInterner<'db>>;
pub type CanonicalVarKind<'db> = rustc_type_ir::CanonicalVarKind<DbInterner<'db>>;
pub type CanonicalQueryInput<'db, V> = rustc_type_ir::CanonicalQueryInput<DbInterner<'db>, V>;
pub type AliasTy<'db> = rustc_type_ir::AliasTy<DbInterner<'db>>;
pub type FnSig<'db> = rustc_type_ir::FnSig<DbInterner<'db>>;
pub type PolyFnSig<'db> = Binder<'db, rustc_type_ir::FnSig<DbInterner<'db>>>;
pub type TypingMode<'db> = rustc_type_ir::TypingMode<DbInterner<'db>>;
pub type TypeError<'db> = rustc_type_ir::error::TypeError<DbInterner<'db>>;
pub type QueryResult<'db> = rustc_type_ir::solve::QueryResult<DbInterner<'db>>;
pub type FxIndexMap<K, V> = rustc_type_ir::data_structures::IndexMap<K, V>;
