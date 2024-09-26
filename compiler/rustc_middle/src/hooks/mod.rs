//! "Hooks" provide a way for `tcx` functionality to be provided by some downstream crate without
//! everything in rustc having to depend on that crate. This is somewhat similar to queries, but
//! queries come with a lot of machinery for caching and incremental compilation, whereas hooks are
//! just plain function pointers without any of the query magic.

use rustc_hir::def_id::{DefId, DefPathHash};
use rustc_session::StableCrateId;
use rustc_span::def_id::{CrateNum, LocalDefId};
use rustc_span::{DUMMY_SP, ExpnHash, ExpnId};
use tracing::instrument;

use crate::mir;
use crate::query::TyCtxtAt;
use crate::ty::{Ty, TyCtxt};

macro_rules! declare_hooks {
    ($($(#[$attr:meta])*hook $name:ident($($arg:ident: $K:ty),*) -> $V:ty;)*) => {

        impl<'tcx> TyCtxt<'tcx> {
            $(
            $(#[$attr])*
            #[inline(always)]
            pub fn $name(self, $($arg: $K,)*) -> $V
            {
                self.at(DUMMY_SP).$name($($arg,)*)
            }
            )*
        }

        impl<'tcx> TyCtxtAt<'tcx> {
            $(
            $(#[$attr])*
            #[inline(always)]
            #[instrument(level = "debug", skip(self), ret)]
            pub fn $name(self, $($arg: $K,)*) -> $V
            {
                (self.tcx.hooks.$name)(self, $($arg,)*)
            }
            )*
        }

        pub struct Providers {
            $(pub $name: for<'tcx> fn(
                TyCtxtAt<'tcx>,
                $($arg: $K,)*
            ) -> $V,)*
        }

        impl Default for Providers {
            fn default() -> Self {
                Providers {
                    $($name: |_, $($arg,)*| default_hook(stringify!($name), &($($arg,)*))),*
                }
            }
        }

        impl Copy for Providers {}
        impl Clone for Providers {
            fn clone(&self) -> Self { *self }
        }
    };
}

declare_hooks! {
    /// Tries to destructure an `mir::Const` ADT or array into its variant index
    /// and its field values. This should only be used for pretty printing.
    hook try_destructure_mir_constant_for_user_output(val: mir::ConstValue<'tcx>, ty: Ty<'tcx>) -> Option<mir::DestructuredConstant<'tcx>>;

    /// Getting a &core::panic::Location referring to a span.
    hook const_caller_location(file: rustc_span::Symbol, line: u32, col: u32) -> mir::ConstValue<'tcx>;

    /// Returns `true` if this def is a function-like thing that is eligible for
    /// coverage instrumentation under `-Cinstrument-coverage`.
    ///
    /// (Eligible functions might nevertheless be skipped for other reasons.)
    hook is_eligible_for_coverage(key: LocalDefId) -> bool;

    /// Create the MIR for a given `DefId` - this includes
    /// unreachable code.
    /// You do not want to call this yourself, instead use the cached version
    /// via `mir_built`
    hook build_mir(key: LocalDefId) -> mir::Body<'tcx>;

    /// Imports all `SourceFile`s from the given crate into the current session.
    /// This normally happens automatically when we decode a `Span` from
    /// that crate's metadata - however, the incr comp cache needs
    /// to trigger this manually when decoding a foreign `Span`
    hook import_source_files(key: CrateNum) -> ();

    hook expn_hash_to_expn_id(
        cnum: CrateNum,
        index_guess: u32,
        hash: ExpnHash
    ) -> ExpnId;

    /// Converts a `DefPathHash` to its corresponding `DefId` in the current compilation
    /// session, if it still exists. This is used during incremental compilation to
    /// turn a deserialized `DefPathHash` into its current `DefId`.
    /// Will fetch a DefId from a DefPathHash for a foreign crate.
    hook def_path_hash_to_def_id_extern(hash: DefPathHash, stable_crate_id: StableCrateId) -> DefId;

    /// Create a THIR tree for debugging.
    hook thir_tree(key: LocalDefId) -> String;

    /// Create a list-like THIR representation for debugging.
    hook thir_flat(key: LocalDefId) -> String;

    /// Returns `true` if we should codegen an instance in the local crate, or returns `false` if we
    /// can just link to the upstream crate and therefore don't need a mono item.
    hook should_codegen_locally(instance: crate::ty::Instance<'tcx>) -> bool;
}

#[cold]
fn default_hook(name: &str, args: &dyn std::fmt::Debug) -> ! {
    bug!(
        "`tcx.{name}{args:?}` cannot be called as `{name}` was never assigned to a provider function"
    )
}
