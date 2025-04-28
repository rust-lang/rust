//! This module contains paths to types and functions Clippy needs to know
//! about.
//!
//! Whenever possible, please consider diagnostic items over hardcoded paths.
//! See <https://github.com/rust-lang/rust-clippy/issues/5393> for more information.

use crate::{MaybePath, PathNS, lookup_path, path_def_id, sym};
use rustc_hir::def_id::DefId;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::{STDLIB_STABLE_CRATES, Symbol};
use std::sync::OnceLock;

/// Lazily resolves a path into a list of [`DefId`]s using [`lookup_path`].
///
/// Typically it will contain one [`DefId`] or none, but in some situations there can be multiple:
/// - `memchr::memchr` could return the functions from both memchr 1.0 and memchr 2.0
/// - `alloc::boxed::Box::downcast` would return a function for each of the different inherent impls
///   ([1], [2], [3])
///
/// [1]: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.downcast
/// [2]: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.downcast-1
/// [3]: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.downcast-2
pub struct PathLookup {
    ns: PathNS,
    path: &'static [Symbol],
    once: OnceLock<Vec<DefId>>,
}

impl PathLookup {
    /// Only exported for tests and `clippy_lints_internal`
    #[doc(hidden)]
    pub const fn new(ns: PathNS, path: &'static [Symbol]) -> Self {
        Self {
            ns,
            path,
            once: OnceLock::new(),
        }
    }

    /// Returns the list of [`DefId`]s that the path resolves to
    pub fn get(&self, cx: &LateContext<'_>) -> &[DefId] {
        self.once.get_or_init(|| lookup_path(cx.tcx, self.ns, self.path))
    }

    /// Returns the single [`DefId`] that the path resolves to, this can only be used for paths into
    /// stdlib crates to avoid the issue of multiple [`DefId`]s being returned
    ///
    /// May return [`None`] in `no_std`/`no_core` environments
    pub fn only(&self, cx: &LateContext<'_>) -> Option<DefId> {
        let ids = self.get(cx);
        debug_assert!(STDLIB_STABLE_CRATES.contains(&self.path[0]));
        debug_assert!(ids.len() <= 1, "{ids:?}");
        ids.first().copied()
    }

    /// Checks if the path resolves to the given `def_id`
    pub fn matches(&self, cx: &LateContext<'_>, def_id: DefId) -> bool {
        self.get(cx).contains(&def_id)
    }

    /// Resolves `maybe_path` to a [`DefId`] and checks if the [`PathLookup`] matches it
    pub fn matches_path<'tcx>(&self, cx: &LateContext<'_>, maybe_path: &impl MaybePath<'tcx>) -> bool {
        path_def_id(cx, maybe_path).is_some_and(|def_id| self.matches(cx, def_id))
    }

    /// Checks if the path resolves to `ty`'s definition, must be an `Adt`
    pub fn matches_ty(&self, cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
        ty.ty_adt_def().is_some_and(|adt| self.matches(cx, adt.did()))
    }
}

macro_rules! path_macros {
    ($($name:ident: $ns:expr,)*) => {
        $(
            /// Only exported for tests and `clippy_lints_internal`
            #[doc(hidden)]
            #[macro_export]
            macro_rules! $name {
                ($$($$seg:ident $$(::)?)*) => {
                    PathLookup::new($ns, &[$$(sym::$$seg,)*])
                };
            }
        )*
    };
}

path_macros! {
    type_path: PathNS::Type,
    value_path: PathNS::Value,
    macro_path: PathNS::Macro,
}

// Paths in `core`/`alloc`/`std`. This should be avoided and cleaned up by adding diagnostic items.
pub static ALIGN_OF: PathLookup = value_path!(core::mem::align_of);
pub static CHAR_TO_DIGIT: PathLookup = value_path!(char::to_digit);
pub static IO_ERROR_NEW: PathLookup = value_path!(std::io::Error::new);
pub static IO_ERRORKIND_OTHER_CTOR: PathLookup = value_path!(std::io::ErrorKind::Other);
pub static ITER_STEP: PathLookup = type_path!(core::iter::Step);
pub static SLICE_FROM_REF: PathLookup = value_path!(core::slice::from_ref);

// Paths in external crates
pub static FUTURES_IO_ASYNCREADEXT: PathLookup = type_path!(futures_util::AsyncReadExt);
pub static FUTURES_IO_ASYNCWRITEEXT: PathLookup = type_path!(futures_util::AsyncWriteExt);
pub static ITERTOOLS_NEXT_TUPLE: PathLookup = value_path!(itertools::Itertools::next_tuple);
pub static PARKING_LOT_GUARDS: [PathLookup; 3] = [
    type_path!(lock_api::mutex::MutexGuard),
    type_path!(lock_api::rwlock::RwLockReadGuard),
    type_path!(lock_api::rwlock::RwLockWriteGuard),
];
pub static REGEX_BUILDER_NEW: PathLookup = value_path!(regex::RegexBuilder::new);
pub static REGEX_BYTES_BUILDER_NEW: PathLookup = value_path!(regex::bytes::RegexBuilder::new);
pub static REGEX_BYTES_NEW: PathLookup = value_path!(regex::bytes::Regex::new);
pub static REGEX_BYTES_SET_NEW: PathLookup = value_path!(regex::bytes::RegexSet::new);
pub static REGEX_NEW: PathLookup = value_path!(regex::Regex::new);
pub static REGEX_SET_NEW: PathLookup = value_path!(regex::RegexSet::new);
pub static SERDE_DESERIALIZE: PathLookup = type_path!(serde::de::Deserialize);
pub static SERDE_DE_VISITOR: PathLookup = type_path!(serde::de::Visitor);
pub static TOKIO_FILE_OPTIONS: PathLookup = value_path!(tokio::fs::File::options);
pub static TOKIO_IO_ASYNCREADEXT: PathLookup = type_path!(tokio::io::AsyncReadExt);
pub static TOKIO_IO_ASYNCWRITEEXT: PathLookup = type_path!(tokio::io::AsyncWriteExt);
pub static TOKIO_IO_OPEN_OPTIONS: PathLookup = type_path!(tokio::fs::OpenOptions);
pub static TOKIO_IO_OPEN_OPTIONS_NEW: PathLookup = value_path!(tokio::fs::OpenOptions::new);
pub static LAZY_STATIC: PathLookup = macro_path!(lazy_static::lazy_static);
pub static ONCE_CELL_SYNC_LAZY: PathLookup = type_path!(once_cell::sync::Lazy);
pub static ONCE_CELL_SYNC_LAZY_NEW: PathLookup = value_path!(once_cell::sync::Lazy::new);

// Paths for internal lints go in `clippy_lints_internal/src/internal_paths.rs`
