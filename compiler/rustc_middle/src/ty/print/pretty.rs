use crate::mir::interpret::{AllocRange, GlobalAlloc, Pointer, Provenance, Scalar};
use crate::query::IntoQueryParam;
use crate::query::Providers;
use crate::ty::{
    self, ConstInt, ParamConst, ScalarInt, Term, TermKind, Ty, TyCtxt, TypeFoldable,
    TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
};
use crate::ty::{GenericArg, GenericArgKind};
use rustc_apfloat::ieee::{Double, Single};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::sso::SsoHashSet;
use rustc_hir as hir;
use rustc_hir::def::{self, CtorKind, DefKind, Namespace};
use rustc_hir::def_id::{DefId, DefIdSet, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPathData, DefPathDataName, DisambiguatedDefPathData};
use rustc_hir::LangItem;
use rustc_session::config::TrimmedDefPaths;
use rustc_session::cstore::{ExternCrate, ExternCrateSource};
use rustc_session::Limit;
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::FileNameDisplayPreference;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;
use smallvec::SmallVec;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::iter;
use std::ops::{ControlFlow, Deref, DerefMut};

// `pretty` is a separate module only for organization.
use super::*;

macro_rules! p {
    (@$lit:literal) => {
        write!(scoped_cx!(), $lit)?
    };
    (@write($($data:expr),+)) => {
        write!(scoped_cx!(), $($data),+)?
    };
    (@print($x:expr)) => {
        scoped_cx!() = $x.print(scoped_cx!())?
    };
    (@$method:ident($($arg:expr),*)) => {
        scoped_cx!() = scoped_cx!().$method($($arg),*)?
    };
    ($($elem:tt $(($($args:tt)*))?),+) => {{
        $(p!(@ $elem $(($($args)*))?);)+
    }};
}
macro_rules! define_scoped_cx {
    ($cx:ident) => {
        #[allow(unused_macros)]
        macro_rules! scoped_cx {
            () => {
                $cx
            };
        }
    };
}

thread_local! {
    static FORCE_IMPL_FILENAME_LINE: Cell<bool> = const { Cell::new(false) };
    static SHOULD_PREFIX_WITH_CRATE: Cell<bool> = const { Cell::new(false) };
    static NO_TRIMMED_PATH: Cell<bool> = const { Cell::new(false) };
    static FORCE_TRIMMED_PATH: Cell<bool> = const { Cell::new(false) };
    static NO_QUERIES: Cell<bool> = const { Cell::new(false) };
    static NO_VISIBLE_PATH: Cell<bool> = const { Cell::new(false) };
}

macro_rules! define_helper {
    ($($(#[$a:meta])* fn $name:ident($helper:ident, $tl:ident);)+) => {
        $(
            #[must_use]
            pub struct $helper(bool);

            impl $helper {
                pub fn new() -> $helper {
                    $helper($tl.with(|c| c.replace(true)))
                }
            }

            $(#[$a])*
            pub macro $name($e:expr) {
                {
                    let _guard = $helper::new();
                    $e
                }
            }

            impl Drop for $helper {
                fn drop(&mut self) {
                    $tl.with(|c| c.set(self.0))
                }
            }
        )+
    }
}

define_helper!(
    /// Avoids running any queries during any prints that occur
    /// during the closure. This may alter the appearance of some
    /// types (e.g. forcing verbose printing for opaque types).
    /// This method is used during some queries (e.g. `explicit_item_bounds`
    /// for opaque types), to ensure that any debug printing that
    /// occurs during the query computation does not end up recursively
    /// calling the same query.
    fn with_no_queries(NoQueriesGuard, NO_QUERIES);
    /// Force us to name impls with just the filename/line number. We
    /// normally try to use types. But at some points, notably while printing
    /// cycle errors, this can result in extra or suboptimal error output,
    /// so this variable disables that check.
    fn with_forced_impl_filename_line(ForcedImplGuard, FORCE_IMPL_FILENAME_LINE);
    /// Adds the `crate::` prefix to paths where appropriate.
    fn with_crate_prefix(CratePrefixGuard, SHOULD_PREFIX_WITH_CRATE);
    /// Prevent path trimming if it is turned on. Path trimming affects `Display` impl
    /// of various rustc types, for example `std::vec::Vec` would be trimmed to `Vec`,
    /// if no other `Vec` is found.
    fn with_no_trimmed_paths(NoTrimmedGuard, NO_TRIMMED_PATH);
    fn with_forced_trimmed_paths(ForceTrimmedGuard, FORCE_TRIMMED_PATH);
    /// Prevent selection of visible paths. `Display` impl of DefId will prefer
    /// visible (public) reexports of types as paths.
    fn with_no_visible_paths(NoVisibleGuard, NO_VISIBLE_PATH);
);

/// The "region highlights" are used to control region printing during
/// specific error messages. When a "region highlight" is enabled, it
/// gives an alternate way to print specific regions. For now, we
/// always print those regions using a number, so something like "`'0`".
///
/// Regions not selected by the region highlight mode are presently
/// unaffected.
#[derive(Copy, Clone)]
pub struct RegionHighlightMode<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// If enabled, when we see the selected region, use "`'N`"
    /// instead of the ordinary behavior.
    highlight_regions: [Option<(ty::Region<'tcx>, usize)>; 3],

    /// If enabled, when printing a "free region" that originated from
    /// the given `ty::BoundRegionKind`, print it as "`'1`". Free regions that would ordinarily
    /// have names print as normal.
    ///
    /// This is used when you have a signature like `fn foo(x: &u32,
    /// y: &'a u32)` and we want to give a name to the region of the
    /// reference `x`.
    highlight_bound_region: Option<(ty::BoundRegionKind, usize)>,
}

impl<'tcx> RegionHighlightMode<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            highlight_regions: Default::default(),
            highlight_bound_region: Default::default(),
        }
    }

    /// If `region` and `number` are both `Some`, invokes
    /// `highlighting_region`.
    pub fn maybe_highlighting_region(
        &mut self,
        region: Option<ty::Region<'tcx>>,
        number: Option<usize>,
    ) {
        if let Some(k) = region {
            if let Some(n) = number {
                self.highlighting_region(k, n);
            }
        }
    }

    /// Highlights the region inference variable `vid` as `'N`.
    pub fn highlighting_region(&mut self, region: ty::Region<'tcx>, number: usize) {
        let num_slots = self.highlight_regions.len();
        let first_avail_slot =
            self.highlight_regions.iter_mut().find(|s| s.is_none()).unwrap_or_else(|| {
                bug!("can only highlight {} placeholders at a time", num_slots,)
            });
        *first_avail_slot = Some((region, number));
    }

    /// Convenience wrapper for `highlighting_region`.
    pub fn highlighting_region_vid(&mut self, vid: ty::RegionVid, number: usize) {
        self.highlighting_region(ty::Region::new_var(self.tcx, vid), number)
    }

    /// Returns `Some(n)` with the number to use for the given region, if any.
    fn region_highlighted(&self, region: ty::Region<'tcx>) -> Option<usize> {
        self.highlight_regions.iter().find_map(|h| match h {
            Some((r, n)) if *r == region => Some(*n),
            _ => None,
        })
    }

    /// Highlight the given bound region.
    /// We can only highlight one bound region at a time. See
    /// the field `highlight_bound_region` for more detailed notes.
    pub fn highlighting_bound_region(&mut self, br: ty::BoundRegionKind, number: usize) {
        assert!(self.highlight_bound_region.is_none());
        self.highlight_bound_region = Some((br, number));
    }
}

/// Trait for printers that pretty-print using `fmt::Write` to the printer.
pub trait PrettyPrinter<'tcx>:
    Printer<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    > + fmt::Write
{
    /// Like `print_def_path` but for value paths.
    fn print_value_path(
        self,
        def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        self.print_def_path(def_id, substs)
    }

    fn in_binder<T>(self, value: &ty::Binder<'tcx, T>) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        value.as_ref().skip_binder().print(self)
    }

    fn wrap_binder<T, F: FnOnce(&T, Self) -> Result<Self, fmt::Error>>(
        self,
        value: &ty::Binder<'tcx, T>,
        f: F,
    ) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        f(value.as_ref().skip_binder(), self)
    }

    /// Prints comma-separated elements.
    fn comma_sep<T>(mut self, mut elems: impl Iterator<Item = T>) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error>,
    {
        if let Some(first) = elems.next() {
            self = first.print(self)?;
            for elem in elems {
                self.write_str(", ")?;
                self = elem.print(self)?;
            }
        }
        Ok(self)
    }

    /// Prints `{f: t}` or `{f as t}` depending on the `cast` argument
    fn typed_value(
        mut self,
        f: impl FnOnce(Self) -> Result<Self, Self::Error>,
        t: impl FnOnce(Self) -> Result<Self, Self::Error>,
        conversion: &str,
    ) -> Result<Self::Const, Self::Error> {
        self.write_str("{")?;
        self = f(self)?;
        self.write_str(conversion)?;
        self = t(self)?;
        self.write_str("}")?;
        Ok(self)
    }

    /// Prints `<...>` around what `f` prints.
    fn generic_delimiters(
        self,
        f: impl FnOnce(Self) -> Result<Self, Self::Error>,
    ) -> Result<Self, Self::Error>;

    /// Returns `true` if the region should be printed in
    /// optional positions, e.g., `&'a T` or `dyn Tr + 'b`.
    /// This is typically the case for all non-`'_` regions.
    fn should_print_region(&self, region: ty::Region<'tcx>) -> bool;

    fn reset_type_limit(&mut self) {}

    // Defaults (should not be overridden):

    /// If possible, this returns a global path resolving to `def_id` that is visible
    /// from at least one local module, and returns `true`. If the crate defining `def_id` is
    /// declared with an `extern crate`, the path is guaranteed to use the `extern crate`.
    fn try_print_visible_def_path(self, def_id: DefId) -> Result<(Self, bool), Self::Error> {
        if NO_VISIBLE_PATH.with(|flag| flag.get()) {
            return Ok((self, false));
        }

        let mut callers = Vec::new();
        self.try_print_visible_def_path_recur(def_id, &mut callers)
    }

    // Given a `DefId`, produce a short name. For types and traits, it prints *only* its name,
    // For associated items on traits it prints out the trait's name and the associated item's name.
    // For enum variants, if they have an unique name, then we only print the name, otherwise we
    // print the enum name and the variant name. Otherwise, we do not print anything and let the
    // caller use the `print_def_path` fallback.
    fn force_print_trimmed_def_path(
        mut self,
        def_id: DefId,
    ) -> Result<(Self::Path, bool), Self::Error> {
        let key = self.tcx().def_key(def_id);
        let visible_parent_map = self.tcx().visible_parent_map(());
        let kind = self.tcx().def_kind(def_id);

        let get_local_name = |this: &Self, name, def_id, key: DefKey| {
            if let Some(visible_parent) = visible_parent_map.get(&def_id)
                && let actual_parent = this.tcx().opt_parent(def_id)
                && let DefPathData::TypeNs(_) = key.disambiguated_data.data
                && Some(*visible_parent) != actual_parent
            {
                this
                    .tcx()
                    .module_children(visible_parent)
                    .iter()
                    .filter(|child| child.res.opt_def_id() == Some(def_id))
                    .find(|child| child.vis.is_public() && child.ident.name != kw::Underscore)
                    .map(|child| child.ident.name)
                    .unwrap_or(name)
            } else {
                name
            }
        };
        if let DefKind::Variant = kind
            && let Some(symbol) = self.tcx().trimmed_def_paths(()).get(&def_id)
        {
            // If `Assoc` is unique, we don't want to talk about `Trait::Assoc`.
            self.write_str(get_local_name(&self, *symbol, def_id, key).as_str())?;
            return Ok((self, true));
        }
        if let Some(symbol) = key.get_opt_name() {
            if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = kind
                && let Some(parent) = self.tcx().opt_parent(def_id)
                && let parent_key = self.tcx().def_key(parent)
                && let Some(symbol) = parent_key.get_opt_name()
            {
                // Trait
                self.write_str(get_local_name(&self, symbol, parent, parent_key).as_str())?;
                self.write_str("::")?;
            } else if let DefKind::Variant = kind
                && let Some(parent) = self.tcx().opt_parent(def_id)
                && let parent_key = self.tcx().def_key(parent)
                && let Some(symbol) = parent_key.get_opt_name()
            {
                // Enum

                // For associated items and variants, we want the "full" path, namely, include
                // the parent type in the path. For example, `Iterator::Item`.
                self.write_str(get_local_name(&self, symbol, parent, parent_key).as_str())?;
                self.write_str("::")?;
            } else if let DefKind::Struct | DefKind::Union | DefKind::Enum | DefKind::Trait
                | DefKind::TyAlias | DefKind::Fn | DefKind::Const | DefKind::Static(_) = kind
            {
            } else {
                // If not covered above, like for example items out of `impl` blocks, fallback.
                return Ok((self, false));
            }
            self.write_str(get_local_name(&self, symbol, def_id, key).as_str())?;
            return Ok((self, true));
        }
        Ok((self, false))
    }

    /// Try to see if this path can be trimmed to a unique symbol name.
    fn try_print_trimmed_def_path(
        mut self,
        def_id: DefId,
    ) -> Result<(Self::Path, bool), Self::Error> {
        if FORCE_TRIMMED_PATH.with(|flag| flag.get()) {
            let (s, trimmed) = self.force_print_trimmed_def_path(def_id)?;
            if trimmed {
                return Ok((s, true));
            }
            self = s;
        }
        if !self.tcx().sess.opts.unstable_opts.trim_diagnostic_paths
            || matches!(self.tcx().sess.opts.trimmed_def_paths, TrimmedDefPaths::Never)
            || NO_TRIMMED_PATH.with(|flag| flag.get())
            || SHOULD_PREFIX_WITH_CRATE.with(|flag| flag.get())
        {
            return Ok((self, false));
        }

        match self.tcx().trimmed_def_paths(()).get(&def_id) {
            None => Ok((self, false)),
            Some(symbol) => {
                write!(self, "{}", Ident::with_dummy_span(*symbol))?;
                Ok((self, true))
            }
        }
    }

    /// Does the work of `try_print_visible_def_path`, building the
    /// full definition path recursively before attempting to
    /// post-process it into the valid and visible version that
    /// accounts for re-exports.
    ///
    /// This method should only be called by itself or
    /// `try_print_visible_def_path`.
    ///
    /// `callers` is a chain of visible_parent's leading to `def_id`,
    /// to support cycle detection during recursion.
    ///
    /// This method returns false if we can't print the visible path, so
    /// `print_def_path` can fall back on the item's real definition path.
    fn try_print_visible_def_path_recur(
        mut self,
        def_id: DefId,
        callers: &mut Vec<DefId>,
    ) -> Result<(Self, bool), Self::Error> {
        define_scoped_cx!(self);

        debug!("try_print_visible_def_path: def_id={:?}", def_id);

        // If `def_id` is a direct or injected extern crate, return the
        // path to the crate followed by the path to the item within the crate.
        if let Some(cnum) = def_id.as_crate_root() {
            if cnum == LOCAL_CRATE {
                return Ok((self.path_crate(cnum)?, true));
            }

            // In local mode, when we encounter a crate other than
            // LOCAL_CRATE, execution proceeds in one of two ways:
            //
            // 1. For a direct dependency, where user added an
            //    `extern crate` manually, we put the `extern
            //    crate` as the parent. So you wind up with
            //    something relative to the current crate.
            // 2. For an extern inferred from a path or an indirect crate,
            //    where there is no explicit `extern crate`, we just prepend
            //    the crate name.
            match self.tcx().extern_crate(def_id) {
                Some(&ExternCrate { src, dependency_of, span, .. }) => match (src, dependency_of) {
                    (ExternCrateSource::Extern(def_id), LOCAL_CRATE) => {
                        // NOTE(eddyb) the only reason `span` might be dummy,
                        // that we're aware of, is that it's the `std`/`core`
                        // `extern crate` injected by default.
                        // FIXME(eddyb) find something better to key this on,
                        // or avoid ending up with `ExternCrateSource::Extern`,
                        // for the injected `std`/`core`.
                        if span.is_dummy() {
                            return Ok((self.path_crate(cnum)?, true));
                        }

                        // Disable `try_print_trimmed_def_path` behavior within
                        // the `print_def_path` call, to avoid infinite recursion
                        // in cases where the `extern crate foo` has non-trivial
                        // parents, e.g. it's nested in `impl foo::Trait for Bar`
                        // (see also issues #55779 and #87932).
                        self = with_no_visible_paths!(self.print_def_path(def_id, &[])?);

                        return Ok((self, true));
                    }
                    (ExternCrateSource::Path, LOCAL_CRATE) => {
                        return Ok((self.path_crate(cnum)?, true));
                    }
                    _ => {}
                },
                None => {
                    return Ok((self.path_crate(cnum)?, true));
                }
            }
        }

        if def_id.is_local() {
            return Ok((self, false));
        }

        let visible_parent_map = self.tcx().visible_parent_map(());

        let mut cur_def_key = self.tcx().def_key(def_id);
        debug!("try_print_visible_def_path: cur_def_key={:?}", cur_def_key);

        // For a constructor, we want the name of its parent rather than <unnamed>.
        if let DefPathData::Ctor = cur_def_key.disambiguated_data.data {
            let parent = DefId {
                krate: def_id.krate,
                index: cur_def_key
                    .parent
                    .expect("`DefPathData::Ctor` / `VariantData` missing a parent"),
            };

            cur_def_key = self.tcx().def_key(parent);
        }

        let Some(visible_parent) = visible_parent_map.get(&def_id).cloned() else {
            return Ok((self, false));
        };

        let actual_parent = self.tcx().opt_parent(def_id);
        debug!(
            "try_print_visible_def_path: visible_parent={:?} actual_parent={:?}",
            visible_parent, actual_parent,
        );

        let mut data = cur_def_key.disambiguated_data.data;
        debug!(
            "try_print_visible_def_path: data={:?} visible_parent={:?} actual_parent={:?}",
            data, visible_parent, actual_parent,
        );

        match data {
            // In order to output a path that could actually be imported (valid and visible),
            // we need to handle re-exports correctly.
            //
            // For example, take `std::os::unix::process::CommandExt`, this trait is actually
            // defined at `std::sys::unix::ext::process::CommandExt` (at time of writing).
            //
            // `std::os::unix` reexports the contents of `std::sys::unix::ext`. `std::sys` is
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
            DefPathData::TypeNs(ref mut name) if Some(visible_parent) != actual_parent => {
                // Item might be re-exported several times, but filter for the one
                // that's public and whose identifier isn't `_`.
                let reexport = self
                    .tcx()
                    .module_children(visible_parent)
                    .iter()
                    .filter(|child| child.res.opt_def_id() == Some(def_id))
                    .find(|child| child.vis.is_public() && child.ident.name != kw::Underscore)
                    .map(|child| child.ident.name);

                if let Some(new_name) = reexport {
                    *name = new_name;
                } else {
                    // There is no name that is public and isn't `_`, so bail.
                    return Ok((self, false));
                }
            }
            // Re-exported `extern crate` (#43189).
            DefPathData::CrateRoot => {
                data = DefPathData::TypeNs(self.tcx().crate_name(def_id.krate));
            }
            _ => {}
        }
        debug!("try_print_visible_def_path: data={:?}", data);

        if callers.contains(&visible_parent) {
            return Ok((self, false));
        }
        callers.push(visible_parent);
        // HACK(eddyb) this bypasses `path_append`'s prefix printing to avoid
        // knowing ahead of time whether the entire path will succeed or not.
        // To support printers that do not implement `PrettyPrinter`, a `Vec` or
        // linked list on the stack would need to be built, before any printing.
        match self.try_print_visible_def_path_recur(visible_parent, callers)? {
            (cx, false) => return Ok((cx, false)),
            (cx, true) => self = cx,
        }
        callers.pop();

        Ok((self.path_append(Ok, &DisambiguatedDefPathData { data, disambiguator: 0 })?, true))
    }

    fn pretty_path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        if trait_ref.is_none() {
            // Inherent impls. Try to print `Foo::bar` for an inherent
            // impl on `Foo`, but fallback to `<Foo>::bar` if self-type is
            // anything other than a simple path.
            match self_ty.kind() {
                ty::Adt(..)
                | ty::Foreign(_)
                | ty::Bool
                | ty::Char
                | ty::Str
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_) => {
                    return self_ty.print(self);
                }

                _ => {}
            }
        }

        self.generic_delimiters(|mut cx| {
            define_scoped_cx!(cx);

            p!(print(self_ty));
            if let Some(trait_ref) = trait_ref {
                p!(" as ", print(trait_ref.print_only_trait_path()));
            }
            Ok(cx)
        })
    }

    fn pretty_path_append_impl(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        self.generic_delimiters(|mut cx| {
            define_scoped_cx!(cx);

            p!("impl ");
            if let Some(trait_ref) = trait_ref {
                p!(print(trait_ref.print_only_trait_path()), " for ");
            }
            p!(print(self_ty));

            Ok(cx)
        })
    }

    fn pretty_print_type(mut self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
        define_scoped_cx!(self);

        match *ty.kind() {
            ty::Bool => p!("bool"),
            ty::Char => p!("char"),
            ty::Int(t) => p!(write("{}", t.name_str())),
            ty::Uint(t) => p!(write("{}", t.name_str())),
            ty::Float(t) => p!(write("{}", t.name_str())),
            ty::RawPtr(ref tm) => {
                p!(write(
                    "*{} ",
                    match tm.mutbl {
                        hir::Mutability::Mut => "mut",
                        hir::Mutability::Not => "const",
                    }
                ));
                p!(print(tm.ty))
            }
            ty::Ref(r, ty, mutbl) => {
                p!("&");
                if self.should_print_region(r) {
                    p!(print(r), " ");
                }
                p!(print(ty::TypeAndMut { ty, mutbl }))
            }
            ty::Never => p!("!"),
            ty::Tuple(ref tys) => {
                p!("(", comma_sep(tys.iter()));
                if tys.len() == 1 {
                    p!(",");
                }
                p!(")")
            }
            ty::FnDef(def_id, substs) => {
                if NO_QUERIES.with(|q| q.get()) {
                    p!(print_def_path(def_id, substs));
                } else {
                    let sig = self.tcx().fn_sig(def_id).subst(self.tcx(), substs);
                    p!(print(sig), " {{", print_value_path(def_id, substs), "}}");
                }
            }
            ty::FnPtr(ref bare_fn) => p!(print(bare_fn)),
            ty::Infer(infer_ty) => {
                if self.should_print_verbose() {
                    p!(write("{:?}", ty.kind()));
                    return Ok(self);
                }

                if let ty::TyVar(ty_vid) = infer_ty {
                    if let Some(name) = self.ty_infer_name(ty_vid) {
                        p!(write("{}", name))
                    } else {
                        p!(write("{}", infer_ty))
                    }
                } else {
                    p!(write("{}", infer_ty))
                }
            }
            ty::Error(_) => p!("{{type error}}"),
            ty::Param(ref param_ty) => p!(print(param_ty)),
            ty::Bound(debruijn, bound_ty) => match bound_ty.kind {
                ty::BoundTyKind::Anon => {
                    rustc_type_ir::debug_bound_var(&mut self, debruijn, bound_ty.var)?
                }
                ty::BoundTyKind::Param(_, s) => match self.should_print_verbose() {
                    true => p!(write("{:?}", ty.kind())),
                    false => p!(write("{s}")),
                },
            },
            ty::Adt(def, substs) => {
                p!(print_def_path(def.did(), substs));
            }
            ty::Dynamic(data, r, repr) => {
                let print_r = self.should_print_region(r);
                if print_r {
                    p!("(");
                }
                match repr {
                    ty::Dyn => p!("dyn "),
                    ty::DynStar => p!("dyn* "),
                }
                p!(print(data));
                if print_r {
                    p!(" + ", print(r), ")");
                }
            }
            ty::Foreign(def_id) => {
                p!(print_def_path(def_id, &[]));
            }
            ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ref data) => {
                if !(self.should_print_verbose() || NO_QUERIES.with(|q| q.get()))
                    && self.tcx().is_impl_trait_in_trait(data.def_id)
                {
                    return self.pretty_print_opaque_impl_type(data.def_id, data.substs);
                } else {
                    p!(print(data))
                }
            }
            ty::Placeholder(placeholder) => match placeholder.bound.kind {
                ty::BoundTyKind::Anon => p!(write("{placeholder:?}")),
                ty::BoundTyKind::Param(_, name) => match self.should_print_verbose() {
                    true => p!(write("{:?}", ty.kind())),
                    false => p!(write("{name}")),
                },
            },
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                // We use verbose printing in 'NO_QUERIES' mode, to
                // avoid needing to call `predicates_of`. This should
                // only affect certain debug messages (e.g. messages printed
                // from `rustc_middle::ty` during the computation of `tcx.predicates_of`),
                // and should have no effect on any compiler output.
                if self.should_print_verbose() {
                    // FIXME(eddyb) print this with `print_def_path`.
                    p!(write("Opaque({:?}, {:?})", def_id, substs));
                    return Ok(self);
                }

                let parent = self.tcx().parent(def_id);
                match self.tcx().def_kind(parent) {
                    DefKind::TyAlias | DefKind::AssocTy => {
                        // NOTE: I know we should check for NO_QUERIES here, but it's alright.
                        // `type_of` on a type alias or assoc type should never cause a cycle.
                        if let ty::Alias(ty::Opaque, ty::AliasTy { def_id: d, .. }) =
                            *self.tcx().type_of(parent).subst_identity().kind()
                        {
                            if d == def_id {
                                // If the type alias directly starts with the `impl` of the
                                // opaque type we're printing, then skip the `::{opaque#1}`.
                                p!(print_def_path(parent, substs));
                                return Ok(self);
                            }
                        }
                        // Complex opaque type, e.g. `type Foo = (i32, impl Debug);`
                        p!(print_def_path(def_id, substs));
                        return Ok(self);
                    }
                    _ => {
                        if NO_QUERIES.with(|q| q.get()) {
                            p!(print_def_path(def_id, &[]));
                            return Ok(self);
                        } else {
                            return self.pretty_print_opaque_impl_type(def_id, substs);
                        }
                    }
                }
            }
            ty::Str => p!("str"),
            ty::Generator(did, substs, movability) => {
                p!(write("["));
                let generator_kind = self.tcx().generator_kind(did).unwrap();
                let should_print_movability =
                    self.should_print_verbose() || generator_kind == hir::GeneratorKind::Gen;

                if should_print_movability {
                    match movability {
                        hir::Movability::Movable => {}
                        hir::Movability::Static => p!("static "),
                    }
                }

                if !self.should_print_verbose() {
                    p!(write("{}", generator_kind));
                    // FIXME(eddyb) should use `def_span`.
                    if let Some(did) = did.as_local() {
                        let span = self.tcx().def_span(did);
                        p!(write(
                            "@{}",
                            // This may end up in stderr diagnostics but it may also be emitted
                            // into MIR. Hence we use the remapped path if available
                            self.tcx().sess.source_map().span_to_embeddable_string(span)
                        ));
                    } else {
                        p!(write("@"), print_def_path(did, substs));
                    }
                } else {
                    p!(print_def_path(did, substs));
                    p!(" upvar_tys=(");
                    if !substs.as_generator().is_valid() {
                        p!("unavailable");
                    } else {
                        self = self.comma_sep(substs.as_generator().upvar_tys())?;
                    }
                    p!(")");

                    if substs.as_generator().is_valid() {
                        p!(" ", print(substs.as_generator().witness()));
                    }
                }

                p!("]")
            }
            ty::GeneratorWitness(types) => {
                p!(in_binder(&types));
            }
            ty::GeneratorWitnessMIR(did, substs) => {
                p!(write("["));
                if !self.tcx().sess.verbose() {
                    p!("generator witness");
                    // FIXME(eddyb) should use `def_span`.
                    if let Some(did) = did.as_local() {
                        let span = self.tcx().def_span(did);
                        p!(write(
                            "@{}",
                            // This may end up in stderr diagnostics but it may also be emitted
                            // into MIR. Hence we use the remapped path if available
                            self.tcx().sess.source_map().span_to_embeddable_string(span)
                        ));
                    } else {
                        p!(write("@"), print_def_path(did, substs));
                    }
                } else {
                    p!(print_def_path(did, substs));
                }

                p!("]")
            }
            ty::Closure(did, substs) => {
                p!(write("["));
                if !self.should_print_verbose() {
                    p!(write("closure"));
                    // FIXME(eddyb) should use `def_span`.
                    if let Some(did) = did.as_local() {
                        if self.tcx().sess.opts.unstable_opts.span_free_formats {
                            p!("@", print_def_path(did.to_def_id(), substs));
                        } else {
                            let span = self.tcx().def_span(did);
                            let preference = if FORCE_TRIMMED_PATH.with(|flag| flag.get()) {
                                FileNameDisplayPreference::Short
                            } else {
                                FileNameDisplayPreference::Remapped
                            };
                            p!(write(
                                "@{}",
                                // This may end up in stderr diagnostics but it may also be emitted
                                // into MIR. Hence we use the remapped path if available
                                self.tcx().sess.source_map().span_to_string(span, preference)
                            ));
                        }
                    } else {
                        p!(write("@"), print_def_path(did, substs));
                    }
                } else {
                    p!(print_def_path(did, substs));
                    if !substs.as_closure().is_valid() {
                        p!(" closure_substs=(unavailable)");
                        p!(write(" substs={:?}", substs));
                    } else {
                        p!(" closure_kind_ty=", print(substs.as_closure().kind_ty()));
                        p!(
                            " closure_sig_as_fn_ptr_ty=",
                            print(substs.as_closure().sig_as_fn_ptr_ty())
                        );
                        p!(" upvar_tys=(");
                        self = self.comma_sep(substs.as_closure().upvar_tys())?;
                        p!(")");
                    }
                }
                p!("]");
            }
            ty::Array(ty, sz) => p!("[", print(ty), "; ", print(sz), "]"),
            ty::Slice(ty) => p!("[", print(ty), "]"),
        }

        Ok(self)
    }

    fn pretty_print_opaque_impl_type(
        mut self,
        def_id: DefId,
        substs: &'tcx ty::List<ty::GenericArg<'tcx>>,
    ) -> Result<Self::Type, Self::Error> {
        let tcx = self.tcx();

        // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
        // by looking up the projections associated with the def_id.
        let bounds = tcx.explicit_item_bounds(def_id);

        let mut traits = FxIndexMap::default();
        let mut fn_traits = FxIndexMap::default();
        let mut is_sized = false;
        let mut lifetimes = SmallVec::<[ty::Region<'tcx>; 1]>::new();

        for (predicate, _) in bounds.subst_iter_copied(tcx, substs) {
            let bound_predicate = predicate.kind();

            match bound_predicate.skip_binder() {
                ty::ClauseKind::Trait(pred) => {
                    let trait_ref = bound_predicate.rebind(pred.trait_ref);

                    // Don't print + Sized, but rather + ?Sized if absent.
                    if Some(trait_ref.def_id()) == tcx.lang_items().sized_trait() {
                        is_sized = true;
                        continue;
                    }

                    self.insert_trait_and_projection(trait_ref, None, &mut traits, &mut fn_traits);
                }
                ty::ClauseKind::Projection(pred) => {
                    let proj_ref = bound_predicate.rebind(pred);
                    let trait_ref = proj_ref.required_poly_trait_ref(tcx);

                    // Projection type entry -- the def-id for naming, and the ty.
                    let proj_ty = (proj_ref.projection_def_id(), proj_ref.term());

                    self.insert_trait_and_projection(
                        trait_ref,
                        Some(proj_ty),
                        &mut traits,
                        &mut fn_traits,
                    );
                }
                ty::ClauseKind::TypeOutlives(outlives) => {
                    lifetimes.push(outlives.1);
                }
                _ => {}
            }
        }

        write!(self, "impl ")?;

        let mut first = true;
        // Insert parenthesis around (Fn(A, B) -> C) if the opaque ty has more than one other trait
        let paren_needed = fn_traits.len() > 1 || traits.len() > 0 || !is_sized;

        for (fn_once_trait_ref, entry) in fn_traits {
            write!(self, "{}", if first { "" } else { " + " })?;
            write!(self, "{}", if paren_needed { "(" } else { "" })?;

            self = self.wrap_binder(&fn_once_trait_ref, |trait_ref, mut cx| {
                define_scoped_cx!(cx);
                // Get the (single) generic ty (the args) of this FnOnce trait ref.
                let generics = tcx.generics_of(trait_ref.def_id);
                let args = generics.own_substs_no_defaults(tcx, trait_ref.substs);

                match (entry.return_ty, args[0].expect_ty()) {
                    // We can only print `impl Fn() -> ()` if we have a tuple of args and we recorded
                    // a return type.
                    (Some(return_ty), arg_tys) if matches!(arg_tys.kind(), ty::Tuple(_)) => {
                        let name = if entry.fn_trait_ref.is_some() {
                            "Fn"
                        } else if entry.fn_mut_trait_ref.is_some() {
                            "FnMut"
                        } else {
                            "FnOnce"
                        };

                        p!(write("{}(", name));

                        for (idx, ty) in arg_tys.tuple_fields().iter().enumerate() {
                            if idx > 0 {
                                p!(", ");
                            }
                            p!(print(ty));
                        }

                        p!(")");
                        if let Some(ty) = return_ty.skip_binder().ty() {
                            if !ty.is_unit() {
                                p!(" -> ", print(return_ty));
                            }
                        }
                        p!(write("{}", if paren_needed { ")" } else { "" }));

                        first = false;
                    }
                    // If we got here, we can't print as a `impl Fn(A, B) -> C`. Just record the
                    // trait_refs we collected in the OpaqueFnEntry as normal trait refs.
                    _ => {
                        if entry.has_fn_once {
                            traits.entry(fn_once_trait_ref).or_default().extend(
                                // Group the return ty with its def id, if we had one.
                                entry
                                    .return_ty
                                    .map(|ty| (tcx.require_lang_item(LangItem::FnOnce, None), ty)),
                            );
                        }
                        if let Some(trait_ref) = entry.fn_mut_trait_ref {
                            traits.entry(trait_ref).or_default();
                        }
                        if let Some(trait_ref) = entry.fn_trait_ref {
                            traits.entry(trait_ref).or_default();
                        }
                    }
                }

                Ok(cx)
            })?;
        }

        // Print the rest of the trait types (that aren't Fn* family of traits)
        for (trait_ref, assoc_items) in traits {
            write!(self, "{}", if first { "" } else { " + " })?;

            self = self.wrap_binder(&trait_ref, |trait_ref, mut cx| {
                define_scoped_cx!(cx);
                p!(print(trait_ref.print_only_trait_name()));

                let generics = tcx.generics_of(trait_ref.def_id);
                let args = generics.own_substs_no_defaults(tcx, trait_ref.substs);

                if !args.is_empty() || !assoc_items.is_empty() {
                    let mut first = true;

                    for ty in args {
                        if first {
                            p!("<");
                            first = false;
                        } else {
                            p!(", ");
                        }
                        p!(print(ty));
                    }

                    for (assoc_item_def_id, term) in assoc_items {
                        // Skip printing `<[generator@] as Generator<_>>::Return` from async blocks,
                        // unless we can find out what generator return type it comes from.
                        let term = if let Some(ty) = term.skip_binder().ty()
                            && let ty::Alias(ty::Projection, proj) = ty.kind()
                            && let Some(assoc) = tcx.opt_associated_item(proj.def_id)
                            && assoc.trait_container(tcx) == tcx.lang_items().gen_trait()
                            && assoc.name == rustc_span::sym::Return
                        {
                            if let ty::Generator(_, substs, _) = substs.type_at(0).kind() {
                                let return_ty = substs.as_generator().return_ty();
                                if !return_ty.is_ty_var() {
                                    return_ty.into()
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        } else {
                            term.skip_binder()
                        };

                        if first {
                            p!("<");
                            first = false;
                        } else {
                            p!(", ");
                        }

                        p!(write("{} = ", tcx.associated_item(assoc_item_def_id).name));

                        match term.unpack() {
                            TermKind::Ty(ty) => p!(print(ty)),
                            TermKind::Const(c) => p!(print(c)),
                        };
                    }

                    if !first {
                        p!(">");
                    }
                }

                first = false;
                Ok(cx)
            })?;
        }

        if !is_sized {
            write!(self, "{}?Sized", if first { "" } else { " + " })?;
        } else if first {
            write!(self, "Sized")?;
        }

        if !FORCE_TRIMMED_PATH.with(|flag| flag.get()) {
            for re in lifetimes {
                write!(self, " + ")?;
                self = self.print_region(re)?;
            }
        }

        Ok(self)
    }

    /// Insert the trait ref and optionally a projection type associated with it into either the
    /// traits map or fn_traits map, depending on if the trait is in the Fn* family of traits.
    fn insert_trait_and_projection(
        &mut self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        proj_ty: Option<(DefId, ty::Binder<'tcx, Term<'tcx>>)>,
        traits: &mut FxIndexMap<
            ty::PolyTraitRef<'tcx>,
            FxIndexMap<DefId, ty::Binder<'tcx, Term<'tcx>>>,
        >,
        fn_traits: &mut FxIndexMap<ty::PolyTraitRef<'tcx>, OpaqueFnEntry<'tcx>>,
    ) {
        let trait_def_id = trait_ref.def_id();

        // If our trait_ref is FnOnce or any of its children, project it onto the parent FnOnce
        // super-trait ref and record it there.
        if let Some(fn_once_trait) = self.tcx().lang_items().fn_once_trait() {
            // If we have a FnOnce, then insert it into
            if trait_def_id == fn_once_trait {
                let entry = fn_traits.entry(trait_ref).or_default();
                // Optionally insert the return_ty as well.
                if let Some((_, ty)) = proj_ty {
                    entry.return_ty = Some(ty);
                }
                entry.has_fn_once = true;
                return;
            } else if Some(trait_def_id) == self.tcx().lang_items().fn_mut_trait() {
                let super_trait_ref = crate::traits::util::supertraits(self.tcx(), trait_ref)
                    .find(|super_trait_ref| super_trait_ref.def_id() == fn_once_trait)
                    .unwrap();

                fn_traits.entry(super_trait_ref).or_default().fn_mut_trait_ref = Some(trait_ref);
                return;
            } else if Some(trait_def_id) == self.tcx().lang_items().fn_trait() {
                let super_trait_ref = crate::traits::util::supertraits(self.tcx(), trait_ref)
                    .find(|super_trait_ref| super_trait_ref.def_id() == fn_once_trait)
                    .unwrap();

                fn_traits.entry(super_trait_ref).or_default().fn_trait_ref = Some(trait_ref);
                return;
            }
        }

        // Otherwise, just group our traits and projection types.
        traits.entry(trait_ref).or_default().extend(proj_ty);
    }

    fn pretty_print_inherent_projection(
        self,
        alias_ty: &ty::AliasTy<'tcx>,
    ) -> Result<Self::Path, Self::Error> {
        let def_key = self.tcx().def_key(alias_ty.def_id);
        self.path_generic_args(
            |cx| {
                cx.path_append(
                    |cx| cx.path_qualified(alias_ty.self_ty(), None),
                    &def_key.disambiguated_data,
                )
            },
            &alias_ty.substs[1..],
        )
    }

    fn ty_infer_name(&self, _: ty::TyVid) -> Option<Symbol> {
        None
    }

    fn const_infer_name(&self, _: ty::ConstVid<'tcx>) -> Option<Symbol> {
        None
    }

    fn pretty_print_dyn_existential(
        mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        // Generate the main trait ref, including associated types.
        let mut first = true;

        if let Some(principal) = predicates.principal() {
            self = self.wrap_binder(&principal, |principal, mut cx| {
                define_scoped_cx!(cx);
                p!(print_def_path(principal.def_id, &[]));

                let mut resugared = false;

                // Special-case `Fn(...) -> ...` and re-sugar it.
                let fn_trait_kind = cx.tcx().fn_trait_kind_from_def_id(principal.def_id);
                if !cx.should_print_verbose() && fn_trait_kind.is_some() {
                    if let ty::Tuple(tys) = principal.substs.type_at(0).kind() {
                        let mut projections = predicates.projection_bounds();
                        if let (Some(proj), None) = (projections.next(), projections.next()) {
                            p!(pretty_fn_sig(
                                tys,
                                false,
                                proj.skip_binder().term.ty().expect("Return type was a const")
                            ));
                            resugared = true;
                        }
                    }
                }

                // HACK(eddyb) this duplicates `FmtPrinter`'s `path_generic_args`,
                // in order to place the projections inside the `<...>`.
                if !resugared {
                    // Use a type that can't appear in defaults of type parameters.
                    let dummy_cx = cx.tcx().mk_fresh_ty(0);
                    let principal = principal.with_self_ty(cx.tcx(), dummy_cx);

                    let args = cx
                        .tcx()
                        .generics_of(principal.def_id)
                        .own_substs_no_defaults(cx.tcx(), principal.substs);

                    let mut projections = predicates.projection_bounds();

                    let mut args = args.iter().cloned();
                    let arg0 = args.next();
                    let projection0 = projections.next();
                    if arg0.is_some() || projection0.is_some() {
                        let args = arg0.into_iter().chain(args);
                        let projections = projection0.into_iter().chain(projections);

                        p!(generic_delimiters(|mut cx| {
                            cx = cx.comma_sep(args)?;
                            if arg0.is_some() && projection0.is_some() {
                                write!(cx, ", ")?;
                            }
                            cx.comma_sep(projections)
                        }));
                    }
                }
                Ok(cx)
            })?;

            first = false;
        }

        define_scoped_cx!(self);

        // Builtin bounds.
        // FIXME(eddyb) avoid printing twice (needed to ensure
        // that the auto traits are sorted *and* printed via cx).
        let mut auto_traits: Vec<_> = predicates.auto_traits().collect();

        // The auto traits come ordered by `DefPathHash`. While
        // `DefPathHash` is *stable* in the sense that it depends on
        // neither the host nor the phase of the moon, it depends
        // "pseudorandomly" on the compiler version and the target.
        //
        // To avoid causing instabilities in compiletest
        // output, sort the auto-traits alphabetically.
        auto_traits.sort_by_cached_key(|did| with_no_trimmed_paths!(self.tcx().def_path_str(*did)));

        for def_id in auto_traits {
            if !first {
                p!(" + ");
            }
            first = false;

            p!(print_def_path(def_id, &[]));
        }

        Ok(self)
    }

    fn pretty_fn_sig(
        mut self,
        inputs: &[Ty<'tcx>],
        c_variadic: bool,
        output: Ty<'tcx>,
    ) -> Result<Self, Self::Error> {
        define_scoped_cx!(self);

        p!("(", comma_sep(inputs.iter().copied()));
        if c_variadic {
            if !inputs.is_empty() {
                p!(", ");
            }
            p!("...");
        }
        p!(")");
        if !output.is_unit() {
            p!(" -> ", print(output));
        }

        Ok(self)
    }

    fn pretty_print_const(
        mut self,
        ct: ty::Const<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        if self.should_print_verbose() {
            p!(write("{:?}", ct));
            return Ok(self);
        }

        macro_rules! print_underscore {
            () => {{
                if print_ty {
                    self = self.typed_value(
                        |mut this| {
                            write!(this, "_")?;
                            Ok(this)
                        },
                        |this| this.print_type(ct.ty()),
                        ": ",
                    )?;
                } else {
                    write!(self, "_")?;
                }
            }};
        }

        match ct.kind() {
            ty::ConstKind::Unevaluated(ty::UnevaluatedConst { def, substs }) => {
                match self.tcx().def_kind(def) {
                    DefKind::Const | DefKind::AssocConst => {
                        p!(print_value_path(def, substs))
                    }
                    DefKind::AnonConst => {
                        if def.is_local()
                            && let span = self.tcx().def_span(def)
                            && let Ok(snip) = self.tcx().sess.source_map().span_to_snippet(span)
                        {
                            p!(write("{}", snip))
                        } else {
                            // Do not call `print_value_path` as if a parent of this anon const is an impl it will
                            // attempt to print out the impl trait ref i.e. `<T as Trait>::{constant#0}`. This would
                            // cause printing to enter an infinite recursion if the anon const is in the self type i.e.
                            // `impl<T: Default> Default for [T; 32 - 1 - 1 - 1] {`
                            // where we would try to print `<[T; /* print `constant#0` again */] as Default>::{constant#0}`
                            p!(write("{}::{}", self.tcx().crate_name(def.krate), self.tcx().def_path(def).to_string_no_crate_verbose()))
                        }
                    }
                    defkind => bug!("`{:?}` has unexpected defkind {:?}", ct, defkind),
                }
            }
            ty::ConstKind::Infer(infer_ct) => {
                match infer_ct {
                    ty::InferConst::Var(ct_vid)
                        if let Some(name) = self.const_infer_name(ct_vid) =>
                            p!(write("{}", name)),
                    _ => print_underscore!(),
                }
            }
            ty::ConstKind::Param(ParamConst { name, .. }) => p!(write("{}", name)),
            ty::ConstKind::Value(value) => {
                return self.pretty_print_const_valtree(value, ct.ty(), print_ty);
            }

            ty::ConstKind::Bound(debruijn, bound_var) => {
                rustc_type_ir::debug_bound_var(&mut self, debruijn, bound_var)?
            }
            ty::ConstKind::Placeholder(placeholder) => p!(write("{placeholder:?}")),
            // FIXME(generic_const_exprs):
            // write out some legible representation of an abstract const?
            ty::ConstKind::Expr(_) => p!("{{const expr}}"),
            ty::ConstKind::Error(_) => p!("{{const error}}"),
        };
        Ok(self)
    }

    fn pretty_print_const_scalar(
        self,
        scalar: Scalar,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        match scalar {
            Scalar::Ptr(ptr, _size) => self.pretty_print_const_scalar_ptr(ptr, ty, print_ty),
            Scalar::Int(int) => self.pretty_print_const_scalar_int(int, ty, print_ty),
        }
    }

    fn pretty_print_const_scalar_ptr(
        mut self,
        ptr: Pointer,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        let (alloc_id, offset) = ptr.into_parts();
        match ty.kind() {
            // Byte strings (&[u8; N])
            ty::Ref(_, inner, _) => {
                if let ty::Array(elem, len) = inner.kind() {
                    if let ty::Uint(ty::UintTy::U8) = elem.kind() {
                        if let ty::ConstKind::Value(ty::ValTree::Leaf(int)) = len.kind() {
                            match self.tcx().try_get_global_alloc(alloc_id) {
                                Some(GlobalAlloc::Memory(alloc)) => {
                                    let len = int.assert_bits(self.tcx().data_layout.pointer_size);
                                    let range =
                                        AllocRange { start: offset, size: Size::from_bytes(len) };
                                    if let Ok(byte_str) =
                                        alloc.inner().get_bytes_strip_provenance(&self.tcx(), range)
                                    {
                                        p!(pretty_print_byte_str(byte_str))
                                    } else {
                                        p!("<too short allocation>")
                                    }
                                }
                                // FIXME: for statics, vtables, and functions, we could in principle print more detail.
                                Some(GlobalAlloc::Static(def_id)) => {
                                    p!(write("<static({:?})>", def_id))
                                }
                                Some(GlobalAlloc::Function(_)) => p!("<function>"),
                                Some(GlobalAlloc::VTable(..)) => p!("<vtable>"),
                                None => p!("<dangling pointer>"),
                            }
                            return Ok(self);
                        }
                    }
                }
            }
            ty::FnPtr(_) => {
                // FIXME: We should probably have a helper method to share code with the "Byte strings"
                // printing above (which also has to handle pointers to all sorts of things).
                if let Some(GlobalAlloc::Function(instance)) =
                    self.tcx().try_get_global_alloc(alloc_id)
                {
                    self = self.typed_value(
                        |this| this.print_value_path(instance.def_id(), instance.substs),
                        |this| this.print_type(ty),
                        " as ",
                    )?;
                    return Ok(self);
                }
            }
            _ => {}
        }
        // Any pointer values not covered by a branch above
        self = self.pretty_print_const_pointer(ptr, ty, print_ty)?;
        Ok(self)
    }

    fn pretty_print_const_scalar_int(
        mut self,
        int: ScalarInt,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        match ty.kind() {
            // Bool
            ty::Bool if int == ScalarInt::FALSE => p!("false"),
            ty::Bool if int == ScalarInt::TRUE => p!("true"),
            // Float
            ty::Float(ty::FloatTy::F32) => {
                p!(write("{}f32", Single::try_from(int).unwrap()))
            }
            ty::Float(ty::FloatTy::F64) => {
                p!(write("{}f64", Double::try_from(int).unwrap()))
            }
            // Int
            ty::Uint(_) | ty::Int(_) => {
                let int =
                    ConstInt::new(int, matches!(ty.kind(), ty::Int(_)), ty.is_ptr_sized_integral());
                if print_ty { p!(write("{:#?}", int)) } else { p!(write("{:?}", int)) }
            }
            // Char
            ty::Char if char::try_from(int).is_ok() => {
                p!(write("{:?}", char::try_from(int).unwrap()))
            }
            // Pointer types
            ty::Ref(..) | ty::RawPtr(_) | ty::FnPtr(_) => {
                let data = int.assert_bits(self.tcx().data_layout.pointer_size);
                self = self.typed_value(
                    |mut this| {
                        write!(this, "0x{:x}", data)?;
                        Ok(this)
                    },
                    |this| this.print_type(ty),
                    " as ",
                )?;
            }
            // Nontrivial types with scalar bit representation
            _ => {
                let print = |mut this: Self| {
                    if int.size() == Size::ZERO {
                        write!(this, "transmute(())")?;
                    } else {
                        write!(this, "transmute(0x{:x})", int)?;
                    }
                    Ok(this)
                };
                self = if print_ty {
                    self.typed_value(print, |this| this.print_type(ty), ": ")?
                } else {
                    print(self)?
                };
            }
        }
        Ok(self)
    }

    /// This is overridden for MIR printing because we only want to hide alloc ids from users, not
    /// from MIR where it is actually useful.
    fn pretty_print_const_pointer<Prov: Provenance>(
        mut self,
        _: Pointer<Prov>,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        if print_ty {
            self.typed_value(
                |mut this| {
                    this.write_str("&_")?;
                    Ok(this)
                },
                |this| this.print_type(ty),
                ": ",
            )
        } else {
            self.write_str("&_")?;
            Ok(self)
        }
    }

    fn pretty_print_byte_str(mut self, byte_str: &'tcx [u8]) -> Result<Self::Const, Self::Error> {
        write!(self, "b\"{}\"", byte_str.escape_ascii())?;
        Ok(self)
    }

    fn pretty_print_const_valtree(
        mut self,
        valtree: ty::ValTree<'tcx>,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        if self.should_print_verbose() {
            p!(write("ValTree({:?}: ", valtree), print(ty), ")");
            return Ok(self);
        }

        let u8_type = self.tcx().types.u8;
        match (valtree, ty.kind()) {
            (ty::ValTree::Branch(_), ty::Ref(_, inner_ty, _)) => match inner_ty.kind() {
                ty::Slice(t) if *t == u8_type => {
                    let bytes = valtree.try_to_raw_bytes(self.tcx(), ty).unwrap_or_else(|| {
                        bug!(
                            "expected to convert valtree {:?} to raw bytes for type {:?}",
                            valtree,
                            t
                        )
                    });
                    return self.pretty_print_byte_str(bytes);
                }
                ty::Str => {
                    let bytes = valtree.try_to_raw_bytes(self.tcx(), ty).unwrap_or_else(|| {
                        bug!("expected to convert valtree to raw bytes for type {:?}", ty)
                    });
                    p!(write("{:?}", String::from_utf8_lossy(bytes)));
                    return Ok(self);
                }
                _ => {
                    p!("&");
                    p!(pretty_print_const_valtree(valtree, *inner_ty, print_ty));
                    return Ok(self);
                }
            },
            (ty::ValTree::Branch(_), ty::Array(t, _)) if *t == u8_type => {
                let bytes = valtree.try_to_raw_bytes(self.tcx(), ty).unwrap_or_else(|| {
                    bug!("expected to convert valtree to raw bytes for type {:?}", t)
                });
                p!("*");
                p!(pretty_print_byte_str(bytes));
                return Ok(self);
            }
            // Aggregates, printed as array/tuple/struct/variant construction syntax.
            (ty::ValTree::Branch(_), ty::Array(..) | ty::Tuple(..) | ty::Adt(..)) => {
                let contents = self.tcx().destructure_const(self.tcx().mk_const(valtree, ty));
                let fields = contents.fields.iter().copied();
                match *ty.kind() {
                    ty::Array(..) => {
                        p!("[", comma_sep(fields), "]");
                    }
                    ty::Tuple(..) => {
                        p!("(", comma_sep(fields));
                        if contents.fields.len() == 1 {
                            p!(",");
                        }
                        p!(")");
                    }
                    ty::Adt(def, _) if def.variants().is_empty() => {
                        self = self.typed_value(
                            |mut this| {
                                write!(this, "unreachable()")?;
                                Ok(this)
                            },
                            |this| this.print_type(ty),
                            ": ",
                        )?;
                    }
                    ty::Adt(def, substs) => {
                        let variant_idx =
                            contents.variant.expect("destructed const of adt without variant idx");
                        let variant_def = &def.variant(variant_idx);
                        p!(print_value_path(variant_def.def_id, substs));
                        match variant_def.ctor_kind() {
                            Some(CtorKind::Const) => {}
                            Some(CtorKind::Fn) => {
                                p!("(", comma_sep(fields), ")");
                            }
                            None => {
                                p!(" {{ ");
                                let mut first = true;
                                for (field_def, field) in iter::zip(&variant_def.fields, fields) {
                                    if !first {
                                        p!(", ");
                                    }
                                    p!(write("{}: ", field_def.name), print(field));
                                    first = false;
                                }
                                p!(" }}");
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                return Ok(self);
            }
            (ty::ValTree::Leaf(leaf), ty::Ref(_, inner_ty, _)) => {
                p!(write("&"));
                return self.pretty_print_const_scalar_int(leaf, *inner_ty, print_ty);
            }
            (ty::ValTree::Leaf(leaf), _) => {
                return self.pretty_print_const_scalar_int(leaf, ty, print_ty);
            }
            // FIXME(oli-obk): also pretty print arrays and other aggregate constants by reading
            // their fields instead of just dumping the memory.
            _ => {}
        }

        // fallback
        if valtree == ty::ValTree::zst() {
            p!(write("<ZST>"));
        } else {
            p!(write("{:?}", valtree));
        }
        if print_ty {
            p!(": ", print(ty));
        }
        Ok(self)
    }

    fn pretty_closure_as_impl(
        mut self,
        closure: ty::ClosureSubsts<'tcx>,
    ) -> Result<Self::Const, Self::Error> {
        let sig = closure.sig();
        let kind = closure.kind_ty().to_opt_closure_kind().unwrap_or(ty::ClosureKind::Fn);

        write!(self, "impl ")?;
        self.wrap_binder(&sig, |sig, mut cx| {
            define_scoped_cx!(cx);

            p!(print(kind), "(");
            for (i, arg) in sig.inputs()[0].tuple_fields().iter().enumerate() {
                if i > 0 {
                    p!(", ");
                }
                p!(print(arg));
            }
            p!(")");

            if !sig.output().is_unit() {
                p!(" -> ", print(sig.output()));
            }

            Ok(cx)
        })
    }

    fn should_print_verbose(&self) -> bool {
        self.tcx().sess.verbose()
    }
}

// HACK(eddyb) boxed to avoid moving around a large struct by-value.
pub struct FmtPrinter<'a, 'tcx>(Box<FmtPrinterData<'a, 'tcx>>);

pub struct FmtPrinterData<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    fmt: String,

    empty_path: bool,
    in_value: bool,
    pub print_alloc_ids: bool,

    // set of all named (non-anonymous) region names
    used_region_names: FxHashSet<Symbol>,

    region_index: usize,
    binder_depth: usize,
    printed_type_count: usize,
    type_length_limit: Limit,
    truncated: bool,

    pub region_highlight_mode: RegionHighlightMode<'tcx>,

    pub ty_infer_name_resolver: Option<Box<dyn Fn(ty::TyVid) -> Option<Symbol> + 'a>>,
    pub const_infer_name_resolver: Option<Box<dyn Fn(ty::ConstVid<'tcx>) -> Option<Symbol> + 'a>>,
}

impl<'a, 'tcx> Deref for FmtPrinter<'a, 'tcx> {
    type Target = FmtPrinterData<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for FmtPrinter<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, 'tcx> FmtPrinter<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, ns: Namespace) -> Self {
        Self::new_with_limit(tcx, ns, tcx.type_length_limit())
    }

    pub fn new_with_limit(tcx: TyCtxt<'tcx>, ns: Namespace, type_length_limit: Limit) -> Self {
        FmtPrinter(Box::new(FmtPrinterData {
            tcx,
            // Estimated reasonable capacity to allocate upfront based on a few
            // benchmarks.
            fmt: String::with_capacity(64),
            empty_path: false,
            in_value: ns == Namespace::ValueNS,
            print_alloc_ids: false,
            used_region_names: Default::default(),
            region_index: 0,
            binder_depth: 0,
            printed_type_count: 0,
            type_length_limit,
            truncated: false,
            region_highlight_mode: RegionHighlightMode::new(tcx),
            ty_infer_name_resolver: None,
            const_infer_name_resolver: None,
        }))
    }

    pub fn into_buffer(self) -> String {
        self.0.fmt
    }
}

// HACK(eddyb) get rid of `def_path_str` and/or pass `Namespace` explicitly always
// (but also some things just print a `DefId` generally so maybe we need this?)
fn guess_def_namespace(tcx: TyCtxt<'_>, def_id: DefId) -> Namespace {
    match tcx.def_key(def_id).disambiguated_data.data {
        DefPathData::TypeNs(..) | DefPathData::CrateRoot | DefPathData::ImplTrait => {
            Namespace::TypeNS
        }

        DefPathData::ValueNs(..)
        | DefPathData::AnonConst
        | DefPathData::ClosureExpr
        | DefPathData::Ctor => Namespace::ValueNS,

        DefPathData::MacroNs(..) => Namespace::MacroNS,

        _ => Namespace::TypeNS,
    }
}

impl<'t> TyCtxt<'t> {
    /// Returns a string identifying this `DefId`. This string is
    /// suitable for user output.
    pub fn def_path_str(self, def_id: impl IntoQueryParam<DefId>) -> String {
        self.def_path_str_with_substs(def_id, &[])
    }

    pub fn def_path_str_with_substs(
        self,
        def_id: impl IntoQueryParam<DefId>,
        substs: &'t [GenericArg<'t>],
    ) -> String {
        let def_id = def_id.into_query_param();
        let ns = guess_def_namespace(self, def_id);
        debug!("def_path_str: def_id={:?}, ns={:?}", def_id, ns);
        FmtPrinter::new(self, ns).print_def_path(def_id, substs).unwrap().into_buffer()
    }

    pub fn value_path_str_with_substs(
        self,
        def_id: impl IntoQueryParam<DefId>,
        substs: &'t [GenericArg<'t>],
    ) -> String {
        let def_id = def_id.into_query_param();
        let ns = guess_def_namespace(self, def_id);
        debug!("value_path_str: def_id={:?}, ns={:?}", def_id, ns);
        FmtPrinter::new(self, ns).print_value_path(def_id, substs).unwrap().into_buffer()
    }
}

impl fmt::Write for FmtPrinter<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.fmt.push_str(s);
        Ok(())
    }
}

impl<'tcx> Printer<'tcx> for FmtPrinter<'_, 'tcx> {
    type Error = fmt::Error;

    type Path = Self;
    type Region = Self;
    type Type = Self;
    type DynExistential = Self;
    type Const = Self;

    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_def_path(
        mut self,
        def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        define_scoped_cx!(self);

        if substs.is_empty() {
            match self.try_print_trimmed_def_path(def_id)? {
                (cx, true) => return Ok(cx),
                (cx, false) => self = cx,
            }

            match self.try_print_visible_def_path(def_id)? {
                (cx, true) => return Ok(cx),
                (cx, false) => self = cx,
            }
        }

        let key = self.tcx.def_key(def_id);
        if let DefPathData::Impl = key.disambiguated_data.data {
            // Always use types for non-local impls, where types are always
            // available, and filename/line-number is mostly uninteresting.
            let use_types = !def_id.is_local() || {
                // Otherwise, use filename/line-number if forced.
                let force_no_types = FORCE_IMPL_FILENAME_LINE.with(|f| f.get());
                !force_no_types
            };

            if !use_types {
                // If no type info is available, fall back to
                // pretty printing some span information. This should
                // only occur very early in the compiler pipeline.
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };
                let span = self.tcx.def_span(def_id);

                self = self.print_def_path(parent_def_id, &[])?;

                // HACK(eddyb) copy of `path_append` to avoid
                // constructing a `DisambiguatedDefPathData`.
                if !self.empty_path {
                    write!(self, "::")?;
                }
                write!(
                    self,
                    "<impl at {}>",
                    // This may end up in stderr diagnostics but it may also be emitted
                    // into MIR. Hence we use the remapped path if available
                    self.tcx.sess.source_map().span_to_embeddable_string(span)
                )?;
                self.empty_path = false;

                return Ok(self);
            }
        }

        self.default_print_def_path(def_id, substs)
    }

    fn print_region(self, region: ty::Region<'tcx>) -> Result<Self::Region, Self::Error> {
        self.pretty_print_region(region)
    }

    fn print_type(mut self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
        if self.type_length_limit.value_within_limit(self.printed_type_count) {
            self.printed_type_count += 1;
            self.pretty_print_type(ty)
        } else {
            self.truncated = true;
            write!(self, "...")?;
            Ok(self)
        }
    }

    fn print_dyn_existential(
        self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        self.pretty_print_dyn_existential(predicates)
    }

    fn print_const(self, ct: ty::Const<'tcx>) -> Result<Self::Const, Self::Error> {
        self.pretty_print_const(ct, false)
    }

    fn path_crate(mut self, cnum: CrateNum) -> Result<Self::Path, Self::Error> {
        self.empty_path = true;
        if cnum == LOCAL_CRATE {
            if self.tcx.sess.rust_2018() {
                // We add the `crate::` keyword on Rust 2018, only when desired.
                if SHOULD_PREFIX_WITH_CRATE.with(|flag| flag.get()) {
                    write!(self, "{}", kw::Crate)?;
                    self.empty_path = false;
                }
            }
        } else {
            write!(self, "{}", self.tcx.crate_name(cnum))?;
            self.empty_path = false;
        }
        Ok(self)
    }

    fn path_qualified(
        mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self = self.pretty_path_qualified(self_ty, trait_ref)?;
        self.empty_path = false;
        Ok(self)
    }

    fn path_append_impl(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        _disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self = self.pretty_path_append_impl(
            |mut cx| {
                cx = print_prefix(cx)?;
                if !cx.empty_path {
                    write!(cx, "::")?;
                }

                Ok(cx)
            },
            self_ty,
            trait_ref,
        )?;
        self.empty_path = false;
        Ok(self)
    }

    fn path_append(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        // Skip `::{{extern}}` blocks and `::{{constructor}}` on tuple/unit structs.
        if let DefPathData::ForeignMod | DefPathData::Ctor = disambiguated_data.data {
            return Ok(self);
        }

        let name = disambiguated_data.data.name();
        if !self.empty_path {
            write!(self, "::")?;
        }

        if let DefPathDataName::Named(name) = name {
            if Ident::with_dummy_span(name).is_raw_guess() {
                write!(self, "r#")?;
            }
        }

        let verbose = self.should_print_verbose();
        disambiguated_data.fmt_maybe_verbose(&mut self, verbose)?;

        self.empty_path = false;

        Ok(self)
    }

    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        if args.first().is_some() {
            if self.in_value {
                write!(self, "::")?;
            }
            self.generic_delimiters(|cx| cx.comma_sep(args.iter().cloned()))
        } else {
            Ok(self)
        }
    }
}

impl<'tcx> PrettyPrinter<'tcx> for FmtPrinter<'_, 'tcx> {
    fn ty_infer_name(&self, id: ty::TyVid) -> Option<Symbol> {
        self.0.ty_infer_name_resolver.as_ref().and_then(|func| func(id))
    }

    fn reset_type_limit(&mut self) {
        self.printed_type_count = 0;
    }

    fn const_infer_name(&self, id: ty::ConstVid<'tcx>) -> Option<Symbol> {
        self.0.const_infer_name_resolver.as_ref().and_then(|func| func(id))
    }

    fn print_value_path(
        mut self,
        def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        let was_in_value = std::mem::replace(&mut self.in_value, true);
        self = self.print_def_path(def_id, substs)?;
        self.in_value = was_in_value;

        Ok(self)
    }

    fn in_binder<T>(self, value: &ty::Binder<'tcx, T>) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        self.pretty_in_binder(value)
    }

    fn wrap_binder<T, C: FnOnce(&T, Self) -> Result<Self, Self::Error>>(
        self,
        value: &ty::Binder<'tcx, T>,
        f: C,
    ) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        self.pretty_wrap_binder(value, f)
    }

    fn typed_value(
        mut self,
        f: impl FnOnce(Self) -> Result<Self, Self::Error>,
        t: impl FnOnce(Self) -> Result<Self, Self::Error>,
        conversion: &str,
    ) -> Result<Self::Const, Self::Error> {
        self.write_str("{")?;
        self = f(self)?;
        self.write_str(conversion)?;
        let was_in_value = std::mem::replace(&mut self.in_value, false);
        self = t(self)?;
        self.in_value = was_in_value;
        self.write_str("}")?;
        Ok(self)
    }

    fn generic_delimiters(
        mut self,
        f: impl FnOnce(Self) -> Result<Self, Self::Error>,
    ) -> Result<Self, Self::Error> {
        write!(self, "<")?;

        let was_in_value = std::mem::replace(&mut self.in_value, false);
        let mut inner = f(self)?;
        inner.in_value = was_in_value;

        write!(inner, ">")?;
        Ok(inner)
    }

    fn should_print_region(&self, region: ty::Region<'tcx>) -> bool {
        let highlight = self.region_highlight_mode;
        if highlight.region_highlighted(region).is_some() {
            return true;
        }

        if self.should_print_verbose() {
            return true;
        }

        if FORCE_TRIMMED_PATH.with(|flag| flag.get()) {
            return false;
        }

        let identify_regions = self.tcx.sess.opts.unstable_opts.identify_regions;

        match *region {
            ty::ReEarlyBound(ref data) => data.has_name(),

            ty::ReLateBound(_, ty::BoundRegion { kind: br, .. })
            | ty::ReFree(ty::FreeRegion { bound_region: br, .. })
            | ty::RePlaceholder(ty::Placeholder {
                bound: ty::BoundRegion { kind: br, .. }, ..
            }) => {
                if br.is_named() {
                    return true;
                }

                if let Some((region, _)) = highlight.highlight_bound_region {
                    if br == region {
                        return true;
                    }
                }

                false
            }

            ty::ReVar(_) if identify_regions => true,

            ty::ReVar(_) | ty::ReErased | ty::ReError(_) => false,

            ty::ReStatic => true,
        }
    }

    fn pretty_print_const_pointer<Prov: Provenance>(
        self,
        p: Pointer<Prov>,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        let print = |mut this: Self| {
            define_scoped_cx!(this);
            if this.print_alloc_ids {
                p!(write("{:?}", p));
            } else {
                p!("&_");
            }
            Ok(this)
        };
        if print_ty {
            self.typed_value(print, |this| this.print_type(ty), ": ")
        } else {
            print(self)
        }
    }
}

// HACK(eddyb) limited to `FmtPrinter` because of `region_highlight_mode`.
impl<'tcx> FmtPrinter<'_, 'tcx> {
    pub fn pretty_print_region(mut self, region: ty::Region<'tcx>) -> Result<Self, fmt::Error> {
        define_scoped_cx!(self);

        // Watch out for region highlights.
        let highlight = self.region_highlight_mode;
        if let Some(n) = highlight.region_highlighted(region) {
            p!(write("'{}", n));
            return Ok(self);
        }

        if self.should_print_verbose() {
            p!(write("{:?}", region));
            return Ok(self);
        }

        let identify_regions = self.tcx.sess.opts.unstable_opts.identify_regions;

        // These printouts are concise. They do not contain all the information
        // the user might want to diagnose an error, but there is basically no way
        // to fit that into a short string. Hence the recommendation to use
        // `explain_region()` or `note_and_explain_region()`.
        match *region {
            ty::ReEarlyBound(ref data) => {
                if data.name != kw::Empty {
                    p!(write("{}", data.name));
                    return Ok(self);
                }
            }
            ty::ReLateBound(_, ty::BoundRegion { kind: br, .. })
            | ty::ReFree(ty::FreeRegion { bound_region: br, .. })
            | ty::RePlaceholder(ty::Placeholder {
                bound: ty::BoundRegion { kind: br, .. }, ..
            }) => {
                if let ty::BrNamed(_, name) = br && br.is_named() {
                    p!(write("{}", name));
                    return Ok(self);
                }

                if let Some((region, counter)) = highlight.highlight_bound_region {
                    if br == region {
                        p!(write("'{}", counter));
                        return Ok(self);
                    }
                }
            }
            ty::ReVar(region_vid) if identify_regions => {
                p!(write("{:?}", region_vid));
                return Ok(self);
            }
            ty::ReVar(_) => {}
            ty::ReErased => {}
            ty::ReError(_) => {}
            ty::ReStatic => {
                p!("'static");
                return Ok(self);
            }
        }

        p!("'_");

        Ok(self)
    }
}

/// Folds through bound vars and placeholders, naming them
struct RegionFolder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    current_index: ty::DebruijnIndex,
    region_map: BTreeMap<ty::BoundRegion, ty::Region<'tcx>>,
    name: &'a mut (
                dyn FnMut(
        Option<ty::DebruijnIndex>, // Debruijn index of the folded late-bound region
        ty::DebruijnIndex,         // Index corresponding to binder level
        ty::BoundRegion,
    ) -> ty::Region<'tcx>
                    + 'a
            ),
}

impl<'a, 'tcx> ty::TypeFolder<TyCtxt<'tcx>> for RegionFolder<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            _ if t.has_vars_bound_at_or_above(self.current_index) || t.has_placeholders() => {
                return t.super_fold_with(self);
            }
            _ => {}
        }
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        let name = &mut self.name;
        let region = match *r {
            ty::ReLateBound(db, br) if db >= self.current_index => {
                *self.region_map.entry(br).or_insert_with(|| name(Some(db), self.current_index, br))
            }
            ty::RePlaceholder(ty::PlaceholderRegion {
                bound: ty::BoundRegion { kind, .. },
                ..
            }) => {
                // If this is an anonymous placeholder, don't rename. Otherwise, in some
                // async fns, we get a `for<'r> Send` bound
                match kind {
                    ty::BrAnon(..) | ty::BrEnv => r,
                    _ => {
                        // Index doesn't matter, since this is just for naming and these never get bound
                        let br = ty::BoundRegion { var: ty::BoundVar::from_u32(0), kind };
                        *self
                            .region_map
                            .entry(br)
                            .or_insert_with(|| name(None, self.current_index, br))
                    }
                }
            }
            _ => return r,
        };
        if let ty::ReLateBound(debruijn1, br) = *region {
            assert_eq!(debruijn1, ty::INNERMOST);
            ty::Region::new_late_bound(self.tcx, self.current_index, br)
        } else {
            region
        }
    }
}

// HACK(eddyb) limited to `FmtPrinter` because of `binder_depth`,
// `region_index` and `used_region_names`.
impl<'tcx> FmtPrinter<'_, 'tcx> {
    pub fn name_all_regions<T>(
        mut self,
        value: &ty::Binder<'tcx, T>,
    ) -> Result<(Self, T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>), fmt::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = fmt::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        fn name_by_region_index(
            index: usize,
            available_names: &mut Vec<Symbol>,
            num_available: usize,
        ) -> Symbol {
            if let Some(name) = available_names.pop() {
                name
            } else {
                Symbol::intern(&format!("'z{}", index - num_available))
            }
        }

        debug!("name_all_regions");

        // Replace any anonymous late-bound regions with named
        // variants, using new unique identifiers, so that we can
        // clearly differentiate between named and unnamed regions in
        // the output. We'll probably want to tweak this over time to
        // decide just how much information to give.
        if self.binder_depth == 0 {
            self.prepare_region_info(value);
        }

        debug!("self.used_region_names: {:?}", &self.used_region_names);

        let mut empty = true;
        let mut start_or_continue = |cx: &mut Self, start: &str, cont: &str| {
            let w = if empty {
                empty = false;
                start
            } else {
                cont
            };
            let _ = write!(cx, "{}", w);
        };
        let do_continue = |cx: &mut Self, cont: Symbol| {
            let _ = write!(cx, "{}", cont);
        };

        define_scoped_cx!(self);

        let possible_names = ('a'..='z').rev().map(|s| Symbol::intern(&format!("'{s}")));

        let mut available_names = possible_names
            .filter(|name| !self.used_region_names.contains(&name))
            .collect::<Vec<_>>();
        debug!(?available_names);
        let num_available = available_names.len();

        let mut region_index = self.region_index;
        let mut next_name = |this: &Self| {
            let mut name;

            loop {
                name = name_by_region_index(region_index, &mut available_names, num_available);
                region_index += 1;

                if !this.used_region_names.contains(&name) {
                    break;
                }
            }

            name
        };

        // If we want to print verbosely, then print *all* binders, even if they
        // aren't named. Eventually, we might just want this as the default, but
        // this is not *quite* right and changes the ordering of some output
        // anyways.
        let (new_value, map) = if self.should_print_verbose() {
            for var in value.bound_vars().iter() {
                start_or_continue(&mut self, "for<", ", ");
                write!(self, "{:?}", var)?;
            }
            start_or_continue(&mut self, "", "> ");
            (value.clone().skip_binder(), BTreeMap::default())
        } else {
            let tcx = self.tcx;

            let trim_path = FORCE_TRIMMED_PATH.with(|flag| flag.get());
            // Closure used in `RegionFolder` to create names for anonymous late-bound
            // regions. We use two `DebruijnIndex`es (one for the currently folded
            // late-bound region and the other for the binder level) to determine
            // whether a name has already been created for the currently folded region,
            // see issue #102392.
            let mut name = |lifetime_idx: Option<ty::DebruijnIndex>,
                            binder_level_idx: ty::DebruijnIndex,
                            br: ty::BoundRegion| {
                let (name, kind) = match br.kind {
                    ty::BrAnon(..) | ty::BrEnv => {
                        let name = next_name(&self);

                        if let Some(lt_idx) = lifetime_idx {
                            if lt_idx > binder_level_idx {
                                let kind = ty::BrNamed(CRATE_DEF_ID.to_def_id(), name);
                                return ty::Region::new_late_bound(
                                    tcx,
                                    ty::INNERMOST,
                                    ty::BoundRegion { var: br.var, kind },
                                );
                            }
                        }

                        (name, ty::BrNamed(CRATE_DEF_ID.to_def_id(), name))
                    }
                    ty::BrNamed(def_id, kw::UnderscoreLifetime | kw::Empty) => {
                        let name = next_name(&self);

                        if let Some(lt_idx) = lifetime_idx {
                            if lt_idx > binder_level_idx {
                                let kind = ty::BrNamed(def_id, name);
                                return ty::Region::new_late_bound(
                                    tcx,
                                    ty::INNERMOST,
                                    ty::BoundRegion { var: br.var, kind },
                                );
                            }
                        }

                        (name, ty::BrNamed(def_id, name))
                    }
                    ty::BrNamed(_, name) => {
                        if let Some(lt_idx) = lifetime_idx {
                            if lt_idx > binder_level_idx {
                                let kind = br.kind;
                                return ty::Region::new_late_bound(
                                    tcx,
                                    ty::INNERMOST,
                                    ty::BoundRegion { var: br.var, kind },
                                );
                            }
                        }

                        (name, br.kind)
                    }
                };

                if !trim_path {
                    start_or_continue(&mut self, "for<", ", ");
                    do_continue(&mut self, name);
                }
                ty::Region::new_late_bound(
                    tcx,
                    ty::INNERMOST,
                    ty::BoundRegion { var: br.var, kind },
                )
            };
            let mut folder = RegionFolder {
                tcx,
                current_index: ty::INNERMOST,
                name: &mut name,
                region_map: BTreeMap::new(),
            };
            let new_value = value.clone().skip_binder().fold_with(&mut folder);
            let region_map = folder.region_map;
            if !trim_path {
                start_or_continue(&mut self, "", "> ");
            }
            (new_value, region_map)
        };

        self.binder_depth += 1;
        self.region_index = region_index;
        Ok((self, new_value, map))
    }

    pub fn pretty_in_binder<T>(self, value: &ty::Binder<'tcx, T>) -> Result<Self, fmt::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = fmt::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        let old_region_index = self.region_index;
        let (new, new_value, _) = self.name_all_regions(value)?;
        let mut inner = new_value.print(new)?;
        inner.region_index = old_region_index;
        inner.binder_depth -= 1;
        Ok(inner)
    }

    pub fn pretty_wrap_binder<T, C: FnOnce(&T, Self) -> Result<Self, fmt::Error>>(
        self,
        value: &ty::Binder<'tcx, T>,
        f: C,
    ) -> Result<Self, fmt::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = fmt::Error> + TypeFoldable<TyCtxt<'tcx>>,
    {
        let old_region_index = self.region_index;
        let (new, new_value, _) = self.name_all_regions(value)?;
        let mut inner = f(&new_value, new)?;
        inner.region_index = old_region_index;
        inner.binder_depth -= 1;
        Ok(inner)
    }

    fn prepare_region_info<T>(&mut self, value: &ty::Binder<'tcx, T>)
    where
        T: TypeVisitable<TyCtxt<'tcx>>,
    {
        struct RegionNameCollector<'tcx> {
            used_region_names: FxHashSet<Symbol>,
            type_collector: SsoHashSet<Ty<'tcx>>,
        }

        impl<'tcx> RegionNameCollector<'tcx> {
            fn new() -> Self {
                RegionNameCollector {
                    used_region_names: Default::default(),
                    type_collector: SsoHashSet::new(),
                }
            }
        }

        impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for RegionNameCollector<'tcx> {
            type BreakTy = ();

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
                trace!("address: {:p}", r.0.0);

                // Collect all named lifetimes. These allow us to prevent duplication
                // of already existing lifetime names when introducing names for
                // anonymous late-bound regions.
                if let Some(name) = r.get_name() {
                    self.used_region_names.insert(name);
                }

                ControlFlow::Continue(())
            }

            // We collect types in order to prevent really large types from compiling for
            // a really long time. See issue #83150 for why this is necessary.
            fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                let not_previously_inserted = self.type_collector.insert(ty);
                if not_previously_inserted {
                    ty.super_visit_with(self)
                } else {
                    ControlFlow::Continue(())
                }
            }
        }

        let mut collector = RegionNameCollector::new();
        value.visit_with(&mut collector);
        self.used_region_names = collector.used_region_names;
        self.region_index = 0;
    }
}

impl<'tcx, T, P: PrettyPrinter<'tcx>> Print<'tcx, P> for ty::Binder<'tcx, T>
where
    T: Print<'tcx, P, Output = P, Error = P::Error> + TypeFoldable<TyCtxt<'tcx>>,
{
    type Output = P;
    type Error = P::Error;

    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.in_binder(self)
    }
}

impl<'tcx, T, U, P: PrettyPrinter<'tcx>> Print<'tcx, P> for ty::OutlivesPredicate<T, U>
where
    T: Print<'tcx, P, Output = P, Error = P::Error>,
    U: Print<'tcx, P, Output = P, Error = P::Error>,
{
    type Output = P;
    type Error = P::Error;
    fn print(&self, mut cx: P) -> Result<Self::Output, Self::Error> {
        define_scoped_cx!(cx);
        p!(print(self.0), ": ", print(self.1));
        Ok(cx)
    }
}

macro_rules! forward_display_to_print {
    ($($ty:ty),+) => {
        // Some of the $ty arguments may not actually use 'tcx
        $(#[allow(unused_lifetimes)] impl<'tcx> fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                ty::tls::with(|tcx| {
                    let cx = tcx.lift(*self)
                        .expect("could not lift for printing")
                        .print(FmtPrinter::new(tcx, Namespace::TypeNS))?;
                    f.write_str(&cx.into_buffer())?;
                    Ok(())
                })
            }
        })+
    };
}

macro_rules! define_print_and_forward_display {
    (($self:ident, $cx:ident): $($ty:ty $print:block)+) => {
        $(impl<'tcx, P: PrettyPrinter<'tcx>> Print<'tcx, P> for $ty {
            type Output = P;
            type Error = fmt::Error;
            fn print(&$self, $cx: P) -> Result<Self::Output, Self::Error> {
                #[allow(unused_mut)]
                let mut $cx = $cx;
                define_scoped_cx!($cx);
                let _: () = $print;
                #[allow(unreachable_code)]
                Ok($cx)
            }
        })+

        forward_display_to_print!($($ty),+);
    };
}

/// Wrapper type for `ty::TraitRef` which opts-in to pretty printing only
/// the trait path. That is, it will print `Trait<U>` instead of
/// `<T as Trait<U>>`.
#[derive(Copy, Clone, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitRefPrintOnlyTraitPath<'tcx>(ty::TraitRef<'tcx>);

impl<'tcx> rustc_errors::IntoDiagnosticArg for TraitRefPrintOnlyTraitPath<'tcx> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        self.to_string().into_diagnostic_arg()
    }
}

impl<'tcx> fmt::Debug for TraitRefPrintOnlyTraitPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Wrapper type for `ty::TraitRef` which opts-in to pretty printing only
/// the trait name. That is, it will print `Trait` instead of
/// `<T as Trait<U>>`.
#[derive(Copy, Clone, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitRefPrintOnlyTraitName<'tcx>(ty::TraitRef<'tcx>);

impl<'tcx> fmt::Debug for TraitRefPrintOnlyTraitName<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<'tcx> ty::TraitRef<'tcx> {
    pub fn print_only_trait_path(self) -> TraitRefPrintOnlyTraitPath<'tcx> {
        TraitRefPrintOnlyTraitPath(self)
    }

    pub fn print_only_trait_name(self) -> TraitRefPrintOnlyTraitName<'tcx> {
        TraitRefPrintOnlyTraitName(self)
    }
}

impl<'tcx> ty::Binder<'tcx, ty::TraitRef<'tcx>> {
    pub fn print_only_trait_path(self) -> ty::Binder<'tcx, TraitRefPrintOnlyTraitPath<'tcx>> {
        self.map_bound(|tr| tr.print_only_trait_path())
    }
}

#[derive(Copy, Clone, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitPredPrintModifiersAndPath<'tcx>(ty::TraitPredicate<'tcx>);

impl<'tcx> fmt::Debug for TraitPredPrintModifiersAndPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<'tcx> ty::TraitPredicate<'tcx> {
    pub fn print_modifiers_and_trait_path(self) -> TraitPredPrintModifiersAndPath<'tcx> {
        TraitPredPrintModifiersAndPath(self)
    }
}

impl<'tcx> ty::PolyTraitPredicate<'tcx> {
    pub fn print_modifiers_and_trait_path(
        self,
    ) -> ty::Binder<'tcx, TraitPredPrintModifiersAndPath<'tcx>> {
        self.map_bound(TraitPredPrintModifiersAndPath)
    }
}

#[derive(Debug, Copy, Clone, Lift)]
pub struct PrintClosureAsImpl<'tcx> {
    pub closure: ty::ClosureSubsts<'tcx>,
}

forward_display_to_print! {
    ty::Region<'tcx>,
    Ty<'tcx>,
    &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ty::Const<'tcx>,

    // HACK(eddyb) these are exhaustive instead of generic,
    // because `for<'tcx>` isn't possible yet.
    ty::PolyExistentialPredicate<'tcx>,
    ty::Binder<'tcx, ty::TraitRef<'tcx>>,
    ty::Binder<'tcx, ty::ExistentialTraitRef<'tcx>>,
    ty::Binder<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    ty::Binder<'tcx, TraitRefPrintOnlyTraitName<'tcx>>,
    ty::Binder<'tcx, ty::FnSig<'tcx>>,
    ty::Binder<'tcx, ty::TraitPredicate<'tcx>>,
    ty::Binder<'tcx, TraitPredPrintModifiersAndPath<'tcx>>,
    ty::Binder<'tcx, ty::SubtypePredicate<'tcx>>,
    ty::Binder<'tcx, ty::ProjectionPredicate<'tcx>>,
    ty::Binder<'tcx, ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>,
    ty::Binder<'tcx, ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>>,

    ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>,
    ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>
}

define_print_and_forward_display! {
    (self, cx):

    &'tcx ty::List<Ty<'tcx>> {
        p!("{{", comma_sep(self.iter()), "}}")
    }

    ty::TypeAndMut<'tcx> {
        p!(write("{}", self.mutbl.prefix_str()), print(self.ty))
    }

    ty::ExistentialTraitRef<'tcx> {
        // Use a type that can't appear in defaults of type parameters.
        let dummy_self = cx.tcx().mk_fresh_ty(0);
        let trait_ref = self.with_self_ty(cx.tcx(), dummy_self);
        p!(print(trait_ref.print_only_trait_path()))
    }

    ty::ExistentialProjection<'tcx> {
        let name = cx.tcx().associated_item(self.def_id).name;
        p!(write("{} = ", name), print(self.term))
    }

    ty::ExistentialPredicate<'tcx> {
        match *self {
            ty::ExistentialPredicate::Trait(x) => p!(print(x)),
            ty::ExistentialPredicate::Projection(x) => p!(print(x)),
            ty::ExistentialPredicate::AutoTrait(def_id) => {
                p!(print_def_path(def_id, &[]));
            }
        }
    }

    ty::FnSig<'tcx> {
        p!(write("{}", self.unsafety.prefix_str()));

        if self.abi != Abi::Rust {
            p!(write("extern {} ", self.abi));
        }

        p!("fn", pretty_fn_sig(self.inputs(), self.c_variadic, self.output()));
    }

    ty::TraitRef<'tcx> {
        p!(write("<{} as {}>", self.self_ty(), self.print_only_trait_path()))
    }

    TraitRefPrintOnlyTraitPath<'tcx> {
        p!(print_def_path(self.0.def_id, self.0.substs));
    }

    TraitRefPrintOnlyTraitName<'tcx> {
        p!(print_def_path(self.0.def_id, &[]));
    }

    TraitPredPrintModifiersAndPath<'tcx> {
        if let ty::BoundConstness::ConstIfConst = self.0.constness {
            p!("~const ")
        }

        if let ty::ImplPolarity::Negative = self.0.polarity {
            p!("!")
        }

        p!(print(self.0.trait_ref.print_only_trait_path()));
    }

    PrintClosureAsImpl<'tcx> {
        p!(pretty_closure_as_impl(self.closure))
    }

    ty::ParamTy {
        p!(write("{}", self.name))
    }

    ty::ParamConst {
        p!(write("{}", self.name))
    }

    ty::SubtypePredicate<'tcx> {
        p!(print(self.a), " <: ");
        cx.reset_type_limit();
        p!(print(self.b))
    }

    ty::CoercePredicate<'tcx> {
        p!(print(self.a), " -> ");
        cx.reset_type_limit();
        p!(print(self.b))
    }

    ty::TraitPredicate<'tcx> {
        p!(print(self.trait_ref.self_ty()), ": ");
        if let ty::BoundConstness::ConstIfConst = self.constness && cx.tcx().features().const_trait_impl {
            p!("~const ");
        }
        if let ty::ImplPolarity::Negative = self.polarity {
            p!("!");
        }
        p!(print(self.trait_ref.print_only_trait_path()))
    }

    ty::ProjectionPredicate<'tcx> {
        p!(print(self.projection_ty), " == ");
        cx.reset_type_limit();
        p!(print(self.term))
    }

    ty::Term<'tcx> {
      match self.unpack() {
        ty::TermKind::Ty(ty) => p!(print(ty)),
        ty::TermKind::Const(c) => p!(print(c)),
      }
    }

    ty::AliasTy<'tcx> {
        if let DefKind::Impl { of_trait: false } = cx.tcx().def_kind(cx.tcx().parent(self.def_id)) {
            p!(pretty_print_inherent_projection(self))
        } else {
            p!(print_def_path(self.def_id, self.substs));
        }
    }

    ty::ClosureKind {
        match *self {
            ty::ClosureKind::Fn => p!("Fn"),
            ty::ClosureKind::FnMut => p!("FnMut"),
            ty::ClosureKind::FnOnce => p!("FnOnce"),
        }
    }

    ty::Predicate<'tcx> {
        let binder = self.kind();
        p!(print(binder))
    }

    ty::PredicateKind<'tcx> {
        match *self {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(ref data)) => {
                p!(print(data))
            }
            ty::PredicateKind::Subtype(predicate) => p!(print(predicate)),
            ty::PredicateKind::Coerce(predicate) => p!(print(predicate)),
            ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(predicate)) => p!(print(predicate)),
            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(predicate)) => p!(print(predicate)),
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(predicate)) => p!(print(predicate)),
            ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
                p!("the constant `", print(ct), "` has type `", print(ty), "`")
            },
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => p!(print(arg), " well-formed"),
            ty::PredicateKind::ObjectSafe(trait_def_id) => {
                p!("the trait `", print_def_path(trait_def_id, &[]), "` is object-safe")
            }
            ty::PredicateKind::ClosureKind(closure_def_id, _closure_substs, kind) => p!(
                "the closure `",
                print_value_path(closure_def_id, &[]),
                write("` implements the trait `{}`", kind)
            ),
            ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(ct)) => {
                p!("the constant `", print(ct), "` can be evaluated")
            }
            ty::PredicateKind::ConstEquate(c1, c2) => {
                p!("the constant `", print(c1), "` equals `", print(c2), "`")
            }
            ty::PredicateKind::TypeWellFormedFromEnv(ty) => {
                p!("the type `", print(ty), "` is found in the environment")
            }
            ty::PredicateKind::Ambiguous => p!("ambiguous"),
            ty::PredicateKind::AliasRelate(t1, t2, dir) => p!(print(t1), write(" {} ", dir), print(t2)),
        }
    }

    GenericArg<'tcx> {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => p!(print(lt)),
            GenericArgKind::Type(ty) => p!(print(ty)),
            GenericArgKind::Const(ct) => p!(print(ct)),
        }
    }
}

fn for_each_def(tcx: TyCtxt<'_>, mut collect_fn: impl for<'b> FnMut(&'b Ident, Namespace, DefId)) {
    // Iterate all local crate items no matter where they are defined.
    let hir = tcx.hir();
    for id in hir.items() {
        if matches!(tcx.def_kind(id.owner_id), DefKind::Use) {
            continue;
        }

        let item = hir.item(id);
        if item.ident.name == kw::Empty {
            continue;
        }

        let def_id = item.owner_id.to_def_id();
        let ns = tcx.def_kind(def_id).ns().unwrap_or(Namespace::TypeNS);
        collect_fn(&item.ident, ns, def_id);
    }

    // Now take care of extern crate items.
    let queue = &mut Vec::new();
    let mut seen_defs: DefIdSet = Default::default();

    for &cnum in tcx.crates(()).iter() {
        let def_id = cnum.as_def_id();

        // Ignore crates that are not direct dependencies.
        match tcx.extern_crate(def_id) {
            None => continue,
            Some(extern_crate) => {
                if !extern_crate.is_direct() {
                    continue;
                }
            }
        }

        queue.push(def_id);
    }

    // Iterate external crate defs but be mindful about visibility
    while let Some(def) = queue.pop() {
        for child in tcx.module_children(def).iter() {
            if !child.vis.is_public() {
                continue;
            }

            match child.res {
                def::Res::Def(DefKind::AssocTy, _) => {}
                def::Res::Def(DefKind::TyAlias, _) => {}
                def::Res::Def(defkind, def_id) => {
                    if let Some(ns) = defkind.ns() {
                        collect_fn(&child.ident, ns, def_id);
                    }

                    if matches!(defkind, DefKind::Mod | DefKind::Enum | DefKind::Trait)
                        && seen_defs.insert(def_id)
                    {
                        queue.push(def_id);
                    }
                }
                _ => {}
            }
        }
    }
}

/// The purpose of this function is to collect public symbols names that are unique across all
/// crates in the build. Later, when printing about types we can use those names instead of the
/// full exported path to them.
///
/// So essentially, if a symbol name can only be imported from one place for a type, and as
/// long as it was not glob-imported anywhere in the current crate, we can trim its printed
/// path and print only the name.
///
/// This has wide implications on error messages with types, for example, shortening
/// `std::vec::Vec` to just `Vec`, as long as there is no other `Vec` importable anywhere.
///
/// The implementation uses similar import discovery logic to that of 'use' suggestions.
///
/// See also [`DelayDm`](rustc_error_messages::DelayDm) and [`with_no_trimmed_paths`].
fn trimmed_def_paths(tcx: TyCtxt<'_>, (): ()) -> FxHashMap<DefId, Symbol> {
    let mut map: FxHashMap<DefId, Symbol> = FxHashMap::default();

    if let TrimmedDefPaths::GoodPath = tcx.sess.opts.trimmed_def_paths {
        // Trimming paths is expensive and not optimized, since we expect it to only be used for error reporting.
        //
        // For good paths causing this bug, the `rustc_middle::ty::print::with_no_trimmed_paths`
        // wrapper can be used to suppress this query, in exchange for full paths being formatted.
        tcx.sess.delay_good_path_bug(
            "trimmed_def_paths constructed but no error emitted; use `DelayDm` for lints or `with_no_trimmed_paths` for debugging",
        );
    }

    let unique_symbols_rev: &mut FxHashMap<(Namespace, Symbol), Option<DefId>> =
        &mut FxHashMap::default();

    for symbol_set in tcx.resolutions(()).glob_map.values() {
        for symbol in symbol_set {
            unique_symbols_rev.insert((Namespace::TypeNS, *symbol), None);
            unique_symbols_rev.insert((Namespace::ValueNS, *symbol), None);
            unique_symbols_rev.insert((Namespace::MacroNS, *symbol), None);
        }
    }

    for_each_def(tcx, |ident, ns, def_id| {
        use std::collections::hash_map::Entry::{Occupied, Vacant};

        match unique_symbols_rev.entry((ns, ident.name)) {
            Occupied(mut v) => match v.get() {
                None => {}
                Some(existing) => {
                    if *existing != def_id {
                        v.insert(None);
                    }
                }
            },
            Vacant(v) => {
                v.insert(Some(def_id));
            }
        }
    });

    for ((_, symbol), opt_def_id) in unique_symbols_rev.drain() {
        use std::collections::hash_map::Entry::{Occupied, Vacant};

        if let Some(def_id) = opt_def_id {
            match map.entry(def_id) {
                Occupied(mut v) => {
                    // A single DefId can be known under multiple names (e.g.,
                    // with a `pub use ... as ...;`). We need to ensure that the
                    // name placed in this map is chosen deterministically, so
                    // if we find multiple names (`symbol`) resolving to the
                    // same `def_id`, we prefer the lexicographically smallest
                    // name.
                    //
                    // Any stable ordering would be fine here though.
                    if *v.get() != symbol {
                        if v.get().as_str() > symbol.as_str() {
                            v.insert(symbol);
                        }
                    }
                }
                Vacant(v) => {
                    v.insert(symbol);
                }
            }
        }
    }

    map
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { trimmed_def_paths, ..*providers };
}

#[derive(Default)]
pub struct OpaqueFnEntry<'tcx> {
    // The trait ref is already stored as a key, so just track if we have it as a real predicate
    has_fn_once: bool,
    fn_mut_trait_ref: Option<ty::PolyTraitRef<'tcx>>,
    fn_trait_ref: Option<ty::PolyTraitRef<'tcx>>,
    return_ty: Option<ty::Binder<'tcx, Term<'tcx>>>,
}
