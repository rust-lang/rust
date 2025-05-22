//! This module contains paths to types and functions Clippy needs to know
//! about.
//!
//! Whenever possible, please consider diagnostic items over hardcoded paths.
//! See <https://github.com/rust-lang/rust-clippy/issues/5393> for more information.

use crate::{MaybePath, path_def_id, sym};
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Namespace::{MacroNS, TypeNS, ValueNS};
use rustc_hir::def::{DefKind, Namespace, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::{ImplItemRef, ItemKind, Node, OwnerId, TraitItemRef, UseKind};
use rustc_lint::LateContext;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{FloatTy, IntTy, Ty, TyCtxt, UintTy};
use rustc_span::{Ident, STDLIB_STABLE_CRATES, Symbol};
use std::sync::OnceLock;

/// Specifies whether to resolve a path in the [`TypeNS`], [`ValueNS`], [`MacroNS`] or in an
/// arbitrary namespace
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PathNS {
    Type,
    Value,
    Macro,

    /// Resolves to the name in the first available namespace, e.g. for `std::vec` this would return
    /// either the macro or the module but **not** both
    ///
    /// Must only be used when the specific resolution is unimportant such as in
    /// `missing_enforced_import_renames`
    Arbitrary,
}

impl PathNS {
    fn matches(self, ns: Option<Namespace>) -> bool {
        let required = match self {
            PathNS::Type => TypeNS,
            PathNS::Value => ValueNS,
            PathNS::Macro => MacroNS,
            PathNS::Arbitrary => return true,
        };

        ns == Some(required)
    }
}

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
pub static CONCAT: PathLookup = macro_path!(core::concat);
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

/// Equivalent to a [`lookup_path`] after splitting the input string on `::`
///
/// This function is expensive and should be used sparingly.
pub fn lookup_path_str(tcx: TyCtxt<'_>, ns: PathNS, path: &str) -> Vec<DefId> {
    let path: Vec<Symbol> = path.split("::").map(Symbol::intern).collect();
    lookup_path(tcx, ns, &path)
}

/// Resolves a def path like `std::vec::Vec`.
///
/// Typically it will return one [`DefId`] or none, but in some situations there can be multiple:
/// - `memchr::memchr` could return the functions from both memchr 1.0 and memchr 2.0
/// - `alloc::boxed::Box::downcast` would return a function for each of the different inherent impls
///   ([1], [2], [3])
///
/// This function is expensive and should be used sparingly.
///
/// [1]: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.downcast
/// [2]: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.downcast-1
/// [3]: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.downcast-2
pub fn lookup_path(tcx: TyCtxt<'_>, ns: PathNS, path: &[Symbol]) -> Vec<DefId> {
    let (root, rest) = match *path {
        [] | [_] => return Vec::new(),
        [root, ref rest @ ..] => (root, rest),
    };

    let mut out = Vec::new();
    for &base in find_crates(tcx, root).iter().chain(find_primitive_impls(tcx, root)) {
        lookup_with_base(tcx, base, ns, rest, &mut out);
    }
    out
}

/// Finds the crates called `name`, may be multiple due to multiple major versions.
pub fn find_crates(tcx: TyCtxt<'_>, name: Symbol) -> &'static [DefId] {
    static BY_NAME: OnceLock<FxHashMap<Symbol, Vec<DefId>>> = OnceLock::new();
    let map = BY_NAME.get_or_init(|| {
        let mut map = FxHashMap::default();
        map.insert(tcx.crate_name(LOCAL_CRATE), vec![LOCAL_CRATE.as_def_id()]);
        for &num in tcx.crates(()) {
            map.entry(tcx.crate_name(num)).or_default().push(num.as_def_id());
        }
        map
    });
    match map.get(&name) {
        Some(def_ids) => def_ids,
        None => &[],
    }
}

fn find_primitive_impls(tcx: TyCtxt<'_>, name: Symbol) -> &[DefId] {
    let ty = match name {
        sym::bool => SimplifiedType::Bool,
        sym::char => SimplifiedType::Char,
        sym::str => SimplifiedType::Str,
        sym::array => SimplifiedType::Array,
        sym::slice => SimplifiedType::Slice,
        // FIXME: rustdoc documents these two using just `pointer`.
        //
        // Maybe this is something we should do here too.
        sym::const_ptr => SimplifiedType::Ptr(Mutability::Not),
        sym::mut_ptr => SimplifiedType::Ptr(Mutability::Mut),
        sym::isize => SimplifiedType::Int(IntTy::Isize),
        sym::i8 => SimplifiedType::Int(IntTy::I8),
        sym::i16 => SimplifiedType::Int(IntTy::I16),
        sym::i32 => SimplifiedType::Int(IntTy::I32),
        sym::i64 => SimplifiedType::Int(IntTy::I64),
        sym::i128 => SimplifiedType::Int(IntTy::I128),
        sym::usize => SimplifiedType::Uint(UintTy::Usize),
        sym::u8 => SimplifiedType::Uint(UintTy::U8),
        sym::u16 => SimplifiedType::Uint(UintTy::U16),
        sym::u32 => SimplifiedType::Uint(UintTy::U32),
        sym::u64 => SimplifiedType::Uint(UintTy::U64),
        sym::u128 => SimplifiedType::Uint(UintTy::U128),
        sym::f32 => SimplifiedType::Float(FloatTy::F32),
        sym::f64 => SimplifiedType::Float(FloatTy::F64),
        _ => return &[],
    };

    tcx.incoherent_impls(ty)
}

/// Resolves a def path like `vec::Vec` with the base `std`.
fn lookup_with_base(tcx: TyCtxt<'_>, mut base: DefId, ns: PathNS, mut path: &[Symbol], out: &mut Vec<DefId>) {
    loop {
        match *path {
            [segment] => {
                out.extend(item_child_by_name(tcx, base, ns, segment));

                // When the current def_id is e.g. `struct S`, check the impl items in
                // `impl S { ... }`
                let inherent_impl_children = tcx
                    .inherent_impls(base)
                    .iter()
                    .filter_map(|&impl_def_id| item_child_by_name(tcx, impl_def_id, ns, segment));
                out.extend(inherent_impl_children);

                return;
            },
            [segment, ref rest @ ..] => {
                path = rest;
                let Some(child) = item_child_by_name(tcx, base, PathNS::Type, segment) else {
                    return;
                };
                base = child;
            },
            [] => unreachable!(),
        }
    }
}

fn item_child_by_name(tcx: TyCtxt<'_>, def_id: DefId, ns: PathNS, name: Symbol) -> Option<DefId> {
    if let Some(local_id) = def_id.as_local() {
        local_item_child_by_name(tcx, local_id, ns, name)
    } else {
        non_local_item_child_by_name(tcx, def_id, ns, name)
    }
}

fn local_item_child_by_name(tcx: TyCtxt<'_>, local_id: LocalDefId, ns: PathNS, name: Symbol) -> Option<DefId> {
    let root_mod;
    let item_kind = match tcx.hir_node_by_def_id(local_id) {
        Node::Crate(r#mod) => {
            root_mod = ItemKind::Mod(Ident::dummy(), r#mod);
            &root_mod
        },
        Node::Item(item) => &item.kind,
        _ => return None,
    };

    let res = |ident: Ident, owner_id: OwnerId| {
        if ident.name == name && ns.matches(tcx.def_kind(owner_id).ns()) {
            Some(owner_id.to_def_id())
        } else {
            None
        }
    };

    match item_kind {
        ItemKind::Mod(_, r#mod) => r#mod.item_ids.iter().find_map(|&item_id| {
            let item = tcx.hir_item(item_id);
            if let ItemKind::Use(path, UseKind::Single(ident)) = item.kind {
                if ident.name == name {
                    path.res
                        .iter()
                        .find(|res| ns.matches(res.ns()))
                        .and_then(Res::opt_def_id)
                } else {
                    None
                }
            } else {
                res(item.kind.ident()?, item_id.owner_id)
            }
        }),
        ItemKind::Impl(r#impl) => r#impl
            .items
            .iter()
            .find_map(|&ImplItemRef { ident, id, .. }| res(ident, id.owner_id)),
        ItemKind::Trait(.., trait_item_refs) => trait_item_refs
            .iter()
            .find_map(|&TraitItemRef { ident, id, .. }| res(ident, id.owner_id)),
        _ => None,
    }
}

fn non_local_item_child_by_name(tcx: TyCtxt<'_>, def_id: DefId, ns: PathNS, name: Symbol) -> Option<DefId> {
    match tcx.def_kind(def_id) {
        DefKind::Mod | DefKind::Enum | DefKind::Trait => tcx.module_children(def_id).iter().find_map(|child| {
            if child.ident.name == name && ns.matches(child.res.ns()) {
                child.res.opt_def_id()
            } else {
                None
            }
        }),
        DefKind::Impl { .. } => tcx
            .associated_item_def_ids(def_id)
            .iter()
            .copied()
            .find(|assoc_def_id| tcx.item_name(*assoc_def_id) == name && ns.matches(tcx.def_kind(assoc_def_id).ns())),
        _ => None,
    }
}
