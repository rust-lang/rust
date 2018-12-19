use crate::hir::def::Namespace;
use crate::hir::map::DefPathData;
use crate::hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use crate::ty::{self, DefIdTree, Ty, TyCtxt};
use crate::ty::print::PrintCx;
use crate::ty::subst::{Subst, SubstsRef};
use crate::middle::cstore::{ExternCrate, ExternCrateSource};
use syntax::ast;
use syntax::symbol::{keywords, Symbol};

use std::cell::Cell;

thread_local! {
    static FORCE_ABSOLUTE: Cell<bool> = Cell::new(false);
    static FORCE_IMPL_FILENAME_LINE: Cell<bool> = Cell::new(false);
    static SHOULD_PREFIX_WITH_CRATE: Cell<bool> = Cell::new(false);
}

/// Enforces that item_path_str always returns an absolute path and
/// also enables "type-based" impl paths. This is used when building
/// symbols that contain types, where we want the crate name to be
/// part of the symbol.
pub fn with_forced_absolute_paths<F: FnOnce() -> R, R>(f: F) -> R {
    FORCE_ABSOLUTE.with(|force| {
        let old = force.get();
        force.set(true);
        let result = f();
        force.set(old);
        result
    })
}

/// Force us to name impls with just the filename/line number. We
/// normally try to use types. But at some points, notably while printing
/// cycle errors, this can result in extra or suboptimal error output,
/// so this variable disables that check.
pub fn with_forced_impl_filename_line<F: FnOnce() -> R, R>(f: F) -> R {
    FORCE_IMPL_FILENAME_LINE.with(|force| {
        let old = force.get();
        force.set(true);
        let result = f();
        force.set(old);
        result
    })
}

/// Adds the `crate::` prefix to paths where appropriate.
pub fn with_crate_prefix<F: FnOnce() -> R, R>(f: F) -> R {
    SHOULD_PREFIX_WITH_CRATE.with(|flag| {
        let old = flag.get();
        flag.set(true);
        let result = f();
        flag.set(old);
        result
    })
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    // HACK(eddyb) get rid of `item_path_str` and/or pass `Namespace` explicitly always
    // (but also some things just print a `DefId` generally so maybe we need this?)
    fn guess_def_namespace(self, def_id: DefId) -> Namespace {
        match self.def_key(def_id).disambiguated_data.data {
            DefPathData::ValueNs(..) |
            DefPathData::EnumVariant(..) |
            DefPathData::Field(..) |
            DefPathData::AnonConst |
            DefPathData::ClosureExpr |
            DefPathData::StructCtor => Namespace::ValueNS,

            DefPathData::MacroDef(..) => Namespace::MacroNS,

            _ => Namespace::TypeNS,
        }
    }

    /// Returns a string identifying this `DefId`. This string is
    /// suitable for user output. It is relative to the current crate
    /// root, unless with_forced_absolute_paths was used.
    pub fn item_path_str_with_substs_and_ns(
        self,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> String {
        debug!("item_path_str: def_id={:?}, substs={:?}, ns={:?}", def_id, substs, ns);
        if FORCE_ABSOLUTE.with(|force| force.get()) {
            PrintCx::new(self, AbsolutePathPrinter).print_item_path(def_id, substs, ns)
        } else {
            PrintCx::new(self, LocalPathPrinter).print_item_path(def_id, substs, ns)
        }
    }

    /// Returns a string identifying this def-id. This string is
    /// suitable for user output. It is relative to the current crate
    /// root, unless with_forced_absolute_paths was used.
    pub fn item_path_str(self, def_id: DefId) -> String {
        let ns = self.guess_def_namespace(def_id);
        self.item_path_str_with_substs_and_ns(def_id, None, ns)
    }

    /// Returns a string identifying this local node-id.
    pub fn node_path_str(self, id: ast::NodeId) -> String {
        self.item_path_str(self.hir().local_def_id(id))
    }

    /// Returns a string identifying this def-id. This string is
    /// suitable for user output. It always begins with a crate identifier.
    pub fn absolute_item_path_str(self, def_id: DefId) -> String {
        debug!("absolute_item_path_str: def_id={:?}", def_id);
        let ns = self.guess_def_namespace(def_id);
        PrintCx::new(self, AbsolutePathPrinter).print_item_path(def_id, None, ns)
    }
}

impl<P: ItemPathPrinter> PrintCx<'a, 'gcx, 'tcx, P> {
    pub fn default_print_item_path(
        &mut self,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> P::Path {
        debug!("default_print_item_path: def_id={:?}, substs={:?}, ns={:?}", def_id, substs, ns);
        let key = self.tcx.def_key(def_id);
        debug!("default_print_item_path: key={:?}", key);
        match key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                assert!(key.parent.is_none());
                self.path_crate(def_id.krate)
            }

            DefPathData::Impl => {
                self.print_impl_path(def_id, substs, ns)
            }

            // Unclear if there is any value in distinguishing these.
            // Probably eventually (and maybe we would even want
            // finer-grained distinctions, e.g., between enum/struct).
            data @ DefPathData::Misc |
            data @ DefPathData::TypeNs(..) |
            data @ DefPathData::Trait(..) |
            data @ DefPathData::TraitAlias(..) |
            data @ DefPathData::AssocTypeInTrait(..) |
            data @ DefPathData::AssocTypeInImpl(..) |
            data @ DefPathData::AssocExistentialInImpl(..) |
            data @ DefPathData::ValueNs(..) |
            data @ DefPathData::Module(..) |
            data @ DefPathData::TypeParam(..) |
            data @ DefPathData::LifetimeParam(..) |
            data @ DefPathData::ConstParam(..) |
            data @ DefPathData::EnumVariant(..) |
            data @ DefPathData::Field(..) |
            data @ DefPathData::AnonConst |
            data @ DefPathData::MacroDef(..) |
            data @ DefPathData::ClosureExpr |
            data @ DefPathData::ImplTrait |
            data @ DefPathData::GlobalMetaData(..) => {
                let parent_did = self.tcx.parent(def_id).unwrap();
                let path = self.print_item_path(parent_did, None, ns);
                self.path_append(path, &data.as_interned_str().as_symbol().as_str())
            },

            DefPathData::StructCtor => { // present `X` instead of `X::{{constructor}}`
                let parent_def_id = self.tcx.parent(def_id).unwrap();
                self.print_item_path(parent_def_id, substs, ns)
            }
        }
    }

    fn default_print_impl_path(
        &mut self,
        impl_def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> P::Path {
        debug!("default_print_impl_path: impl_def_id={:?}", impl_def_id);
        let parent_def_id = self.tcx.parent(impl_def_id).unwrap();

        // Decide whether to print the parent path for the impl.
        // Logically, since impls are global, it's never needed, but
        // users may find it useful. Currently, we omit the parent if
        // the impl is either in the same module as the self-type or
        // as the trait.
        let mut self_ty = self.tcx.type_of(impl_def_id);
        if let Some(substs) = substs {
            self_ty = self_ty.subst(self.tcx, substs);
        }
        let in_self_mod = match characteristic_def_id_of_type(self_ty) {
            None => false,
            Some(ty_def_id) => self.tcx.parent(ty_def_id) == Some(parent_def_id),
        };

        let mut impl_trait_ref = self.tcx.impl_trait_ref(impl_def_id);
        if let Some(substs) = substs {
            impl_trait_ref = impl_trait_ref.subst(self.tcx, substs);
        }
        let in_trait_mod = match impl_trait_ref {
            None => false,
            Some(trait_ref) => self.tcx.parent(trait_ref.def_id) == Some(parent_def_id),
        };

        if !in_self_mod && !in_trait_mod {
            // If the impl is not co-located with either self-type or
            // trait-type, then fallback to a format that identifies
            // the module more clearly.
            let path = self.print_item_path(parent_def_id, None, ns);
            if let Some(trait_ref) = impl_trait_ref {
                return self.path_append(path, &format!("<impl {} for {}>", trait_ref, self_ty));
            } else {
                return self.path_append(path, &format!("<impl {}>", self_ty));
            }
        }

        // Otherwise, try to give a good form that would be valid language
        // syntax. Preferably using associated item notation.

        if let Some(trait_ref) = impl_trait_ref {
            // Trait impls.
            return self.path_impl(&format!("<{} as {}>", self_ty, trait_ref));
        }

        // Inherent impls. Try to print `Foo::bar` for an inherent
        // impl on `Foo`, but fallback to `<Foo>::bar` if self-type is
        // anything other than a simple path.
        match self_ty.sty {
            ty::Adt(adt_def, substs) => {
                // FIXME(eddyb) this should recurse to build the path piecewise.
                // self.print_item_path(adt_def.did, Some(substs), ns)
                let mut s = String::new();
                crate::util::ppaux::parameterized(&mut s, adt_def.did, substs, ns).unwrap();
                self.path_impl(&s)
            }

            ty::Foreign(did) => self.print_item_path(did, None, ns),

            ty::Bool |
            ty::Char |
            ty::Int(_) |
            ty::Uint(_) |
            ty::Float(_) |
            ty::Str => {
                self.path_impl(&self_ty.to_string())
            }

            _ => {
                self.path_impl(&format!("<{}>", self_ty))
            }
        }
    }
}

/// As a heuristic, when we see an impl, if we see that the
/// 'self type' is a type defined in the same module as the impl,
/// we can omit including the path to the impl itself. This
/// function tries to find a "characteristic `DefId`" for a
/// type. It's just a heuristic so it makes some questionable
/// decisions and we may want to adjust it later.
pub fn characteristic_def_id_of_type(ty: Ty<'_>) -> Option<DefId> {
    match ty.sty {
        ty::Adt(adt_def, _) => Some(adt_def.did),

        ty::Dynamic(data, ..) => data.principal_def_id(),

        ty::Array(subty, _) |
        ty::Slice(subty) => characteristic_def_id_of_type(subty),

        ty::RawPtr(mt) => characteristic_def_id_of_type(mt.ty),

        ty::Ref(_, ty, _) => characteristic_def_id_of_type(ty),

        ty::Tuple(ref tys) => tys.iter()
                                   .filter_map(|ty| characteristic_def_id_of_type(ty))
                                   .next(),

        ty::FnDef(def_id, _) |
        ty::Closure(def_id, _) |
        ty::Generator(def_id, _, _) |
        ty::Foreign(def_id) => Some(def_id),

        ty::Bool |
        ty::Char |
        ty::Int(_) |
        ty::Uint(_) |
        ty::Str |
        ty::FnPtr(_) |
        ty::Projection(_) |
        ty::Placeholder(..) |
        ty::UnnormalizedProjection(..) |
        ty::Param(_) |
        ty::Opaque(..) |
        ty::Infer(_) |
        ty::Bound(..) |
        ty::Error |
        ty::GeneratorWitness(..) |
        ty::Never |
        ty::Float(_) => None,
    }
}

/// Unifying Trait for different kinds of item paths we might
/// construct. The basic interface is that components get appended.
pub trait ItemPathPrinter: Sized {
    type Path;

    fn print_item_path(
        self: &mut PrintCx<'_, '_, 'tcx, Self>,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> Self::Path {
        self.default_print_item_path(def_id, substs, ns)
    }
    fn print_impl_path(
        self: &mut PrintCx<'_, '_, 'tcx, Self>,
        impl_def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> Self::Path {
        self.default_print_impl_path(impl_def_id, substs, ns)
    }

    fn path_crate(self: &mut PrintCx<'_, '_, '_, Self>, cnum: CrateNum) -> Self::Path;
    fn path_impl(self: &mut PrintCx<'_, '_, '_, Self>, text: &str) -> Self::Path;
    fn path_append(
        self: &mut PrintCx<'_, '_, '_, Self>,
        path: Self::Path,
        text: &str,
    ) -> Self::Path;
}

struct AbsolutePathPrinter;

impl ItemPathPrinter for AbsolutePathPrinter {
    type Path = String;

    fn path_crate(self: &mut PrintCx<'_, '_, '_, Self>, cnum: CrateNum) -> Self::Path {
        self.tcx.original_crate_name(cnum).to_string()
    }
    fn path_impl(self: &mut PrintCx<'_, '_, '_, Self>, text: &str) -> Self::Path {
        text.to_string()
    }
    fn path_append(
        self: &mut PrintCx<'_, '_, '_, Self>,
        mut path: Self::Path,
        text: &str,
    ) -> Self::Path {
        if !path.is_empty() {
            path.push_str("::");
        }
        path.push_str(text);
        path
    }
}

struct LocalPathPrinter;

impl LocalPathPrinter {
    /// If possible, this returns a global path resolving to `def_id` that is visible
    /// from at least one local module and returns true. If the crate defining `def_id` is
    /// declared with an `extern crate`, the path is guaranteed to use the `extern crate`.
    fn try_print_visible_item_path(
        self: &mut PrintCx<'_, '_, '_, Self>,
        def_id: DefId,
        ns: Namespace,
    ) -> Option<<Self as ItemPathPrinter>::Path> {
        debug!("try_print_visible_item_path: def_id={:?}", def_id);

        // If `def_id` is a direct or injected extern crate, return the
        // path to the crate followed by the path to the item within the crate.
        if def_id.index == CRATE_DEF_INDEX {
            let cnum = def_id.krate;

            if cnum == LOCAL_CRATE {
                return Some(self.path_crate(cnum));
            }

            // In local mode, when we encounter a crate other than
            // LOCAL_CRATE, execution proceeds in one of two ways:
            //
            // 1. for a direct dependency, where user added an
            //    `extern crate` manually, we put the `extern
            //    crate` as the parent. So you wind up with
            //    something relative to the current crate.
            // 2. for an extern inferred from a path or an indirect crate,
            //    where there is no explicit `extern crate`, we just prepend
            //    the crate name.
            match *self.tcx.extern_crate(def_id) {
                Some(ExternCrate {
                    src: ExternCrateSource::Extern(def_id),
                    direct: true,
                    span,
                    ..
                }) => {
                    debug!("try_print_visible_item_path: def_id={:?}", def_id);
                    let path = if !span.is_dummy() {
                        self.print_item_path(def_id, None, ns)
                    } else {
                        self.path_crate(cnum)
                    };
                    return Some(path);
                }
                None => {
                    return Some(self.path_crate(cnum));
                }
                _ => {},
            }
        }

        if def_id.is_local() {
            return None;
        }

        let visible_parent_map = self.tcx.visible_parent_map(LOCAL_CRATE);

        let mut cur_def_key = self.tcx.def_key(def_id);
        debug!("try_print_visible_item_path: cur_def_key={:?}", cur_def_key);

        // For a UnitStruct or TupleStruct we want the name of its parent rather than <unnamed>.
        if let DefPathData::StructCtor = cur_def_key.disambiguated_data.data {
            let parent = DefId {
                krate: def_id.krate,
                index: cur_def_key.parent.expect("DefPathData::StructCtor missing a parent"),
            };

            cur_def_key = self.tcx.def_key(parent);
        }

        let visible_parent = visible_parent_map.get(&def_id).cloned()?;
        let path = self.try_print_visible_item_path(visible_parent, ns)?;
        let actual_parent = self.tcx.parent(def_id);

        let data = cur_def_key.disambiguated_data.data;
        debug!(
            "try_print_visible_item_path: data={:?} visible_parent={:?} actual_parent={:?}",
            data, visible_parent, actual_parent,
        );

        let symbol = match data {
            // In order to output a path that could actually be imported (valid and visible),
            // we need to handle re-exports correctly.
            //
            // For example, take `std::os::unix::process::CommandExt`, this trait is actually
            // defined at `std::sys::unix::ext::process::CommandExt` (at time of writing).
            //
            // `std::os::unix` rexports the contents of `std::sys::unix::ext`. `std::sys` is
            // private so the "true" path to `CommandExt` isn't accessible.
            //
            // In this case, the `visible_parent_map` will look something like this:
            //
            // (child) -> (parent)
            // `std::sys::unix::ext::process::CommandExt` -> `std::sys::unix::ext::process`
            // `std::sys::unix::ext::process` -> `std::sys::unix::ext`
            // `std::sys::unix::ext` -> `std::os`
            //
            // This is correct, as the visible parent of `std::sys::unix::ext` is in fact
            // `std::os`.
            //
            // When printing the path to `CommandExt` and looking at the `cur_def_key` that
            // corresponds to `std::sys::unix::ext`, we would normally print `ext` and then go
            // to the parent - resulting in a mangled path like
            // `std::os::ext::process::CommandExt`.
            //
            // Instead, we must detect that there was a re-export and instead print `unix`
            // (which is the name `std::sys::unix::ext` was re-exported as in `std::os`). To
            // do this, we compare the parent of `std::sys::unix::ext` (`std::sys::unix`) with
            // the visible parent (`std::os`). If these do not match, then we iterate over
            // the children of the visible parent (as was done when computing
            // `visible_parent_map`), looking for the specific child we currently have and then
            // have access to the re-exported name.
            DefPathData::Module(actual_name) |
            DefPathData::TypeNs(actual_name) if Some(visible_parent) != actual_parent => {
                self.tcx.item_children(visible_parent)
                    .iter()
                    .find(|child| child.def.def_id() == def_id)
                    .map(|child| child.ident.as_str())
                    .unwrap_or_else(|| actual_name.as_str())
            }
            _ => {
                data.get_opt_name().map(|n| n.as_str()).unwrap_or_else(|| {
                    // Re-exported `extern crate` (#43189).
                    if let DefPathData::CrateRoot = data {
                        self.tcx.original_crate_name(def_id.krate).as_str()
                    } else {
                        Symbol::intern("<unnamed>").as_str()
                    }
                })
            },
        };
        debug!("try_print_visible_item_path: symbol={:?}", symbol);
        Some(self.path_append(path, &symbol))
    }
}

impl ItemPathPrinter for LocalPathPrinter {
    type Path = String;

    fn print_item_path(
        self: &mut PrintCx<'_, '_, 'tcx, Self>,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> Self::Path {
        self.try_print_visible_item_path(def_id, ns)
            .unwrap_or_else(|| self.default_print_item_path(def_id, substs, ns))
    }
    fn print_impl_path(
        self: &mut PrintCx<'_, '_, 'tcx, Self>,
        impl_def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        ns: Namespace,
    ) -> Self::Path {
        // Always use types for non-local impls, where types are always
        // available, and filename/line-number is mostly uninteresting.
        let use_types = !impl_def_id.is_local() || {
            // Otherwise, use filename/line-number if forced.
            let force_no_types = FORCE_IMPL_FILENAME_LINE.with(|f| f.get());
            !force_no_types
        };

        if !use_types {
            // If no type info is available, fall back to
            // pretty printing some span information. This should
            // only occur very early in the compiler pipeline.
            // FIXME(eddyb) this should just be using `tcx.def_span(impl_def_id)`
            let parent_def_id = self.tcx.parent(impl_def_id).unwrap();
            let path = self.print_item_path(parent_def_id, None, ns);
            let span = self.tcx.def_span(impl_def_id);
            return self.path_append(path, &format!("<impl at {:?}>", span));
        }

        self.default_print_impl_path(impl_def_id, substs, ns)
    }

    fn path_crate(self: &mut PrintCx<'_, '_, '_, Self>, cnum: CrateNum) -> Self::Path {
        if cnum == LOCAL_CRATE {
            if self.tcx.sess.rust_2018() {
                // We add the `crate::` keyword on Rust 2018, only when desired.
                if SHOULD_PREFIX_WITH_CRATE.with(|flag| flag.get()) {
                    return keywords::Crate.name().to_string();
                }
            }
            String::new()
        } else {
            self.tcx.crate_name(cnum).to_string()
        }
    }
    fn path_impl(self: &mut PrintCx<'_, '_, '_, Self>, text: &str) -> Self::Path {
        text.to_string()
    }
    fn path_append(
        self: &mut PrintCx<'_, '_, '_, Self>,
        mut path: Self::Path,
        text: &str,
    ) -> Self::Path {
        if !path.is_empty() {
            path.push_str("::");
        }
        path.push_str(text);
        path
    }
}
