// Aliases
pub use rustc_middle::mir;

// Traits:
pub use super::rustc_extention::{BodyMagic, LocalMagic, PlaceMagic};
pub use itertools::Itertools;
pub use rustc_lint::LateLintPass;
pub use rustc_middle::mir::visit::Visitor;

// Data Structures
pub use rustc_data_structures::fx::{FxHashMap, FxHashSet};
pub use rustc_index::bit_set::BitSet;
pub use rustc_index::IndexVec;
pub use smallvec::SmallVec;
pub use std::collections::{BTreeMap, BTreeSet};

// Common Types
pub use super::{AnalysisInfo, LocalKind, Validity};
pub use rustc_ast::Mutability;
pub use rustc_hir::def_id::{DefId, LocalDefId};
pub use rustc_middle::mir::{
    BasicBlock, BasicBlockData, BorrowKind, Local, Location, Operand, Place, PlaceElem, Rvalue, Statement,
    StatementKind, Terminator, TerminatorKind, VarDebugInfo, VarDebugInfoContents,
};
pub use rustc_middle::ty::TyCtxt;
pub use rustc_span::{sym, Symbol};

// Consts
pub use rustc_middle::mir::START_BLOCK;
pub const RETURN_LOCAL: Local = Local::from_u32(0);
