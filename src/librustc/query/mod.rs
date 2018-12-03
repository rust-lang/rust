use crate::ty::query::QueryDescription;
use crate::ty::query::queries;
use crate::ty::TyCtxt;
use crate::hir::def_id::CrateNum;
use crate::dep_graph::SerializedDepNodeIndex;
use std::borrow::Cow;

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
rustc_queries! {
    Other {
        /// Records the type of every item.
        query type_of(key: DefId) -> Ty<'tcx> {
            cache { key.is_local() }
        }

        query native_libraries(_: CrateNum) -> Lrc<Vec<NativeLibrary>> {
            desc { "looking up the native libraries of a linked crate" }
        }
    }

    Codegen {
        query is_panic_runtime(_: CrateNum) -> bool {
            fatal_cycle
            desc { "checking if the crate is_panic_runtime" }
        }
    }
}
