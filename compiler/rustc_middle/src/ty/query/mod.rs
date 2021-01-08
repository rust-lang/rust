use crate::dep_graph;
use crate::hir::exports::Export;
use crate::hir::map;
use crate::infer::canonical::{self, Canonical};
use crate::lint::LintLevelMap;
use crate::middle::codegen_fn_attrs::CodegenFnAttrs;
use crate::middle::cstore::{CrateDepKind, CrateSource};
use crate::middle::cstore::{ExternCrate, ForeignModule, LinkagePreference, NativeLib};
use crate::middle::exported_symbols::{ExportedSymbol, SymbolExportLevel};
use crate::middle::lib_features::LibFeatures;
use crate::middle::privacy::AccessLevels;
use crate::middle::region;
use crate::middle::resolve_lifetime::{ObjectLifetimeDefault, Region, ResolveLifetimes};
use crate::middle::stability::{self, DeprecationEntry};
use crate::mir;
use crate::mir::interpret::GlobalId;
use crate::mir::interpret::{ConstValue, EvalToAllocationRawResult, EvalToConstValueResult};
use crate::mir::interpret::{LitToConstError, LitToConstInput};
use crate::mir::mono::CodegenUnit;
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
use crate::traits::{self, ImplSource};
use crate::ty::subst::{GenericArg, SubstsRef};
use crate::ty::util::AlwaysRequiresDrop;
use crate::ty::{self, AdtSizedConstraint, CrateInherentImpls, ParamEnvAnd, Ty, TyCtxt};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::stable_hasher::StableVec;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::Lrc;
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefIdSet, LocalDefId};
use rustc_hir::lang_items::{LangItem, LanguageItems};
use rustc_hir::{Crate, ItemLocalId, TraitCandidate};
use rustc_index::{bit_set::FiniteBitSet, vec::IndexVec};
use rustc_session::config::{EntryFnType, OptLevel, OutputFilenames, SymbolManglingVersion};
use rustc_session::utils::NativeLibKind;
use rustc_session::CrateDisambiguator;
use rustc_target::spec::PanicStrategy;

use rustc_ast as ast;
use rustc_attr as attr;
use rustc_span::symbol::Symbol;
use rustc_span::{Span, DUMMY_SP};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

#[macro_use]
mod plumbing;
pub(crate) use rustc_query_system::query::CycleError;
use rustc_query_system::query::*;

mod stats;
pub use self::stats::print_stats;

#[cfg(parallel_compiler)]
mod job;
#[cfg(parallel_compiler)]
pub use self::job::handle_deadlock;
pub use rustc_query_system::query::{QueryInfo, QueryJob, QueryJobId};

mod keys;
use self::keys::Key;

mod values;
use self::values::Value;

use rustc_query_system::query::QueryAccessors;
pub use rustc_query_system::query::QueryConfig;
pub(crate) use rustc_query_system::query::QueryDescription;

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

rustc_query_append! { [define_queries!][<'tcx>] }

mod sealed {
    use super::{DefId, LocalDefId};

    /// An analogue of the `Into` trait that's intended only for query paramaters.
    ///
    /// This exists to allow queries to accept either `DefId` or `LocalDefId` while requiring that the
    /// user call `to_def_id` to convert between them everywhere else.
    pub trait IntoQueryParam<P> {
        fn into_query_param(self) -> P;
    }

    impl<P> IntoQueryParam<P> for P {
        #[inline(always)]
        fn into_query_param(self) -> P {
            self
        }
    }

    impl IntoQueryParam<DefId> for LocalDefId {
        #[inline(always)]
        fn into_query_param(self) -> DefId {
            self.to_def_id()
        }
    }
}

use sealed::IntoQueryParam;
