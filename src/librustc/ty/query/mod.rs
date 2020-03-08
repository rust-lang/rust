use crate::dep_graph::{self, DepConstructor, DepNode};
use crate::hir::exports::Export;
use crate::infer::canonical::{self, Canonical};
use crate::lint::LintLevelMap;
use crate::middle::codegen_fn_attrs::CodegenFnAttrs;
use crate::middle::cstore::{CrateSource, DepKind, NativeLibraryKind};
use crate::middle::cstore::{ExternCrate, ForeignModule, LinkagePreference, NativeLibrary};
use crate::middle::exported_symbols::{ExportedSymbol, SymbolExportLevel};
use crate::middle::lang_items::{LangItem, LanguageItems};
use crate::middle::lib_features::LibFeatures;
use crate::middle::privacy::AccessLevels;
use crate::middle::region;
use crate::middle::resolve_lifetime::{ObjectLifetimeDefault, Region, ResolveLifetimes};
use crate::middle::stability::{self, DeprecationEntry};
use crate::mir;
use crate::mir::interpret::GlobalId;
use crate::mir::interpret::{ConstEvalRawResult, ConstEvalResult, ConstValue};
use crate::mir::interpret::{LitToConstError, LitToConstInput};
use crate::mir::mono::CodegenUnit;
use crate::session::config::{EntryFnType, OptLevel, OutputFilenames, SymbolManglingVersion};
use crate::session::CrateDisambiguator;
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal, CanonicalTyGoal,
    CanonicalTypeOpAscribeUserTypeGoal, CanonicalTypeOpEqGoal, CanonicalTypeOpNormalizeGoal,
    CanonicalTypeOpProvePredicateGoal, CanonicalTypeOpSubtypeGoal, NoSolution,
};
use crate::traits::query::{
    DropckOutlivesResult, DtorckConstraint, MethodAutoderefStepsResult, NormalizationResult,
    OutlivesBound,
};
use crate::traits::specialization_graph;
use crate::traits::Clauses;
use crate::traits::{self, Vtable};
use crate::ty::steal::Steal;
use crate::ty::subst::SubstsRef;
use crate::ty::util::AlwaysRequiresDrop;
use crate::ty::{self, AdtSizedConstraint, CrateInherentImpls, ParamEnvAnd, Ty, TyCtxt};
use crate::util::common::ErrorReported;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::profiling::ProfileCategory::*;
use rustc_data_structures::stable_hasher::StableVec;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefIdSet, DefIndex};
use rustc_hir::{Crate, HirIdSet, ItemLocalId, TraitCandidate};
use rustc_index::vec::IndexVec;
use rustc_target::spec::PanicStrategy;

use rustc_ast::ast;
use rustc_attr as attr;
use rustc_span::symbol::Symbol;
use rustc_span::{Span, DUMMY_SP};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::ops::Deref;
use std::sync::Arc;

#[macro_use]
mod plumbing;
use self::plumbing::*;
pub use self::plumbing::{force_from_dep_node, CycleError};

mod stats;
pub use self::stats::print_stats;

mod job;
#[cfg(parallel_compiler)]
pub use self::job::handle_deadlock;
use self::job::QueryJobInfo;
pub use self::job::{QueryInfo, QueryJob, QueryJobId};

mod keys;
use self::keys::Key;

mod values;
use self::values::Value;

mod caches;
use self::caches::CacheSelector;

mod config;
use self::config::QueryAccessors;
pub use self::config::QueryConfig;
pub(crate) use self::config::QueryDescription;

mod on_disk_cache;
pub use self::on_disk_cache::OnDiskCache;

mod profiling_support;
pub use self::profiling_support::{IntoSelfProfilingString, QueryKeyStringBuilder};

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
