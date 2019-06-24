use crate::dep_graph::{self, DepNode};
use crate::hir::def_id::{CrateNum, DefId, DefIndex};
use crate::hir::def::{DefKind, Export};
use crate::hir::{self, TraitCandidate, ItemLocalId, CodegenFnAttrs};
use crate::infer::canonical::{self, Canonical};
use crate::lint;
use crate::middle::borrowck::BorrowCheckResult;
use crate::middle::cstore::{ExternCrate, LinkagePreference, NativeLibrary, ForeignModule};
use crate::middle::cstore::{NativeLibraryKind, DepKind, CrateSource};
use crate::middle::privacy::AccessLevels;
use crate::middle::reachable::ReachableSet;
use crate::middle::region;
use crate::middle::resolve_lifetime::{ResolveLifetimes, Region, ObjectLifetimeDefault};
use crate::middle::stability::{self, DeprecationEntry};
use crate::middle::lib_features::LibFeatures;
use crate::middle::lang_items::{LanguageItems, LangItem};
use crate::middle::exported_symbols::{SymbolExportLevel, ExportedSymbol};
use crate::mir::interpret::{ConstEvalRawResult, ConstEvalResult};
use crate::mir::mono::CodegenUnit;
use crate::mir;
use crate::mir::interpret::GlobalId;
use crate::session::CrateDisambiguator;
use crate::session::config::{EntryFnType, OutputFilenames, OptLevel, SymbolManglingVersion};
use crate::traits::{self, Vtable};
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal,
    CanonicalTyGoal, CanonicalTypeOpAscribeUserTypeGoal,
    CanonicalTypeOpEqGoal, CanonicalTypeOpSubtypeGoal, CanonicalTypeOpProvePredicateGoal,
    CanonicalTypeOpNormalizeGoal, NoSolution,
};
use crate::traits::query::method_autoderef::MethodAutoderefStepsResult;
use crate::traits::query::dropck_outlives::{DtorckConstraint, DropckOutlivesResult};
use crate::traits::query::normalize::NormalizationResult;
use crate::traits::query::outlives_bounds::OutlivesBound;
use crate::traits::specialization_graph;
use crate::traits::Clauses;
use crate::ty::{self, CrateInherentImpls, ParamEnvAnd, Ty, TyCtxt, AdtSizedConstraint};
use crate::ty::steal::Steal;
use crate::ty::util::NeedsDrop;
use crate::ty::subst::SubstsRef;
use crate::util::nodemap::{DefIdSet, DefIdMap, ItemLocalSet};
use crate::util::common::{ErrorReported};
use crate::util::profiling::ProfileCategory::*;

use rustc_data_structures::svh::Svh;
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::{FxIndexMap, FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::StableVec;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_target::spec::PanicStrategy;

use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Arc;
use std::intrinsics::type_name;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::symbol::InternedString;
use syntax::attr;
use syntax::ast;
use syntax::feature_gate;
use syntax::symbol::Symbol;

#[macro_use]
mod plumbing;
use self::plumbing::*;
pub use self::plumbing::{force_from_dep_node, CycleError};

mod job;
pub use self::job::{QueryJob, QueryInfo};
#[cfg(parallel_compiler)]
pub use self::job::handle_deadlock;

mod keys;
use self::keys::Key;

mod values;
use self::values::Value;

mod config;
pub(crate) use self::config::QueryDescription;
pub use self::config::QueryConfig;
use self::config::QueryAccessors;

mod on_disk_cache;
pub use self::on_disk_cache::OnDiskCache;

// Each of these queries corresponds to a function pointer field in the
// `Providers` struct for requesting a value of that type, and a method
// on `tcx: TyCtxt` (and `tcx.at(span)`) for doing that request in a way
// which memoizes and does dep-graph tracking, wrapping around the actual
// `Providers` that the driver creates (using several `rustc_*` crates).
//
// The result type of each query must implement `Clone`, and additionally
// `ty::query::values::Value`, which produces an appropriate placeholder
// (error) value if the query resulted in a query cycle.
// Queries marked with `fatal_cycle` do not need the latter implementation,
// as they will raise an fatal error on query cycles instead.

rustc_query_append! { [define_queries!][ <'tcx>
    Other {
        /// Runs analysis passes on the crate.
        [eval_always] fn analysis: Analysis(CrateNum) -> Result<(), ErrorReported>,
    },
]}
