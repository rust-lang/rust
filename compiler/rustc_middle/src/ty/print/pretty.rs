use crate::middle::cstore::{ExternCrate, ExternCrateSource};
use crate::mir::interpret::{AllocId, ConstValue, GlobalAlloc, Pointer, Scalar};
use crate::ty::layout::IntegerExt;
use crate::ty::subst::{GenericArg, GenericArgKind, Subst};
use crate::ty::{self, ConstInt, DefIdTree, ParamConst, Ty, TyCtxt, TypeFoldable};
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
use rustc_ast as ast;
use rustc_attr::{SignedInt, UnsignedInt};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::{self, CtorKind, DefKind, Namespace};
use rustc_hir::def_id::{CrateNum, DefId, DefIdSet, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::{DefPathData, DefPathDataName, DisambiguatedDefPathData};
use rustc_hir::ItemKind;
use rustc_session::config::TrimmedDefPaths;
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_target::abi::{Integer, Size};
use rustc_target::spec::abi::Abi;

use std::cell::Cell;
use std::char;
use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
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
    static FORCE_IMPL_FILENAME_LINE: Cell<bool> = Cell::new(false);
    static SHOULD_PREFIX_WITH_CRATE: Cell<bool> = Cell::new(false);
    static NO_TRIMMED_PATH: Cell<bool> = Cell::new(false);
    static NO_QUERIES: Cell<bool> = Cell::new(false);
}

/// Avoids running any queries during any prints that occur
/// during the closure. This may alter the appearance of some
/// types (e.g. forcing verbose printing for opaque types).
/// This method is used during some queries (e.g. `predicates_of`
/// for opaque types), to ensure that any debug printing that
/// occurs during the query computation does not end up recursively
/// calling the same query.
pub fn with_no_queries<F: FnOnce() -> R, R>(f: F) -> R {
    NO_QUERIES.with(|no_queries| {
        let old = no_queries.replace(true);
        let result = f();
        no_queries.set(old);
        result
    })
}

/// Force us to name impls with just the filename/line number. We
/// normally try to use types. But at some points, notably while printing
/// cycle errors, this can result in extra or suboptimal error output,
/// so this variable disables that check.
pub fn with_forced_impl_filename_line<F: FnOnce() -> R, R>(f: F) -> R {
    FORCE_IMPL_FILENAME_LINE.with(|force| {
        let old = force.replace(true);
        let result = f();
        force.set(old);
        result
    })
}

/// Adds the `crate::` prefix to paths where appropriate.
pub fn with_crate_prefix<F: FnOnce() -> R, R>(f: F) -> R {
    SHOULD_PREFIX_WITH_CRATE.with(|flag| {
        let old = flag.replace(true);
        let result = f();
        flag.set(old);
        result
    })
}

/// Prevent path trimming if it is turned on. Path trimming affects `Display` impl
/// of various rustc types, for example `std::vec::Vec` would be trimmed to `Vec`,
/// if no other `Vec` is found.
pub fn with_no_trimmed_paths<F: FnOnce() -> R, R>(f: F) -> R {
    NO_TRIMMED_PATH.with(|flag| {
        let old = flag.replace(true);
        let result = f();
        flag.set(old);
        result
    })
}

/// The "region highlights" are used to control region printing during
/// specific error messages. When a "region highlight" is enabled, it
/// gives an alternate way to print specific regions. For now, we
/// always print those regions using a number, so something like "`'0`".
///
/// Regions not selected by the region highlight mode are presently
/// unaffected.
#[derive(Copy, Clone, Default)]
pub struct RegionHighlightMode {
    /// If enabled, when we see the selected region, use "`'N`"
    /// instead of the ordinary behavior.
    highlight_regions: [Option<(ty::RegionKind, usize)>; 3],

    /// If enabled, when printing a "free region" that originated from
    /// the given `ty::BoundRegion`, print it as "`'1`". Free regions that would ordinarily
    /// have names print as normal.
    ///
    /// This is used when you have a signature like `fn foo(x: &u32,
    /// y: &'a u32)` and we want to give a name to the region of the
    /// reference `x`.
    highlight_bound_region: Option<(ty::BoundRegion, usize)>,
}

impl RegionHighlightMode {
    /// If `region` and `number` are both `Some`, invokes
    /// `highlighting_region`.
    pub fn maybe_highlighting_region(
        &mut self,
        region: Option<ty::Region<'_>>,
        number: Option<usize>,
    ) {
        if let Some(k) = region {
            if let Some(n) = number {
                self.highlighting_region(k, n);
            }
        }
    }

    /// Highlights the region inference variable `vid` as `'N`.
    pub fn highlighting_region(&mut self, region: ty::Region<'_>, number: usize) {
        let num_slots = self.highlight_regions.len();
        let first_avail_slot =
            self.highlight_regions.iter_mut().find(|s| s.is_none()).unwrap_or_else(|| {
                bug!("can only highlight {} placeholders at a time", num_slots,)
            });
        *first_avail_slot = Some((*region, number));
    }

    /// Convenience wrapper for `highlighting_region`.
    pub fn highlighting_region_vid(&mut self, vid: ty::RegionVid, number: usize) {
        self.highlighting_region(&ty::ReVar(vid), number)
    }

    /// Returns `Some(n)` with the number to use for the given region, if any.
    fn region_highlighted(&self, region: ty::Region<'_>) -> Option<usize> {
        self.highlight_regions.iter().find_map(|h| match h {
            Some((r, n)) if r == region => Some(*n),
            _ => None,
        })
    }

    /// Highlight the given bound region.
    /// We can only highlight one bound region at a time. See
    /// the field `highlight_bound_region` for more detailed notes.
    pub fn highlighting_bound_region(&mut self, br: ty::BoundRegion, number: usize) {
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

    fn in_binder<T>(self, value: &ty::Binder<T>) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error> + TypeFoldable<'tcx>,
    {
        value.as_ref().skip_binder().print(self)
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
    fn region_should_not_be_omitted(&self, region: ty::Region<'_>) -> bool;

    // Defaults (should not be overridden):

    /// If possible, this returns a global path resolving to `def_id` that is visible
    /// from at least one local module, and returns `true`. If the crate defining `def_id` is
    /// declared with an `extern crate`, the path is guaranteed to use the `extern crate`.
    fn try_print_visible_def_path(self, def_id: DefId) -> Result<(Self, bool), Self::Error> {
        let mut callers = Vec::new();
        self.try_print_visible_def_path_recur(def_id, &mut callers)
    }

    /// Try to see if this path can be trimmed to a unique symbol name.
    fn try_print_trimmed_def_path(
        mut self,
        def_id: DefId,
    ) -> Result<(Self::Path, bool), Self::Error> {
        if !self.tcx().sess.opts.debugging_opts.trim_diagnostic_paths
            || matches!(self.tcx().sess.opts.trimmed_def_paths, TrimmedDefPaths::Never)
            || NO_TRIMMED_PATH.with(|flag| flag.get())
            || SHOULD_PREFIX_WITH_CRATE.with(|flag| flag.get())
        {
            return Ok((self, false));
        }

        match self.tcx().trimmed_def_paths(LOCAL_CRATE).get(&def_id) {
            None => Ok((self, false)),
            Some(symbol) => {
                self.write_str(&symbol.as_str())?;
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
    fn try_print_visible_def_path_recur(
        mut self,
        def_id: DefId,
        callers: &mut Vec<DefId>,
    ) -> Result<(Self, bool), Self::Error> {
        define_scoped_cx!(self);

        debug!("try_print_visible_def_path: def_id={:?}", def_id);

        // If `def_id` is a direct or injected extern crate, return the
        // path to the crate followed by the path to the item within the crate.
        if def_id.index == CRATE_DEF_INDEX {
            let cnum = def_id.krate;

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
                        debug!("try_print_visible_def_path: def_id={:?}", def_id);
                        return Ok((
                            if !span.is_dummy() {
                                self.print_def_path(def_id, &[])?
                            } else {
                                self.path_crate(cnum)?
                            },
                            true,
                        ));
                    }
                    (ExternCrateSource::Path, LOCAL_CRATE) => {
                        debug!("try_print_visible_def_path: def_id={:?}", def_id);
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

        let visible_parent_map = self.tcx().visible_parent_map(LOCAL_CRATE);

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

        let visible_parent = match visible_parent_map.get(&def_id).cloned() {
            Some(parent) => parent,
            None => return Ok((self, false)),
        };
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
        let actual_parent = self.tcx().parent(def_id);
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
            DefPathData::TypeNs(ref mut name) if Some(visible_parent) != actual_parent => {
                let reexport = self
                    .tcx()
                    .item_children(visible_parent)
                    .iter()
                    .find(|child| child.res.opt_def_id() == Some(def_id))
                    .map(|child| child.ident.name);
                if let Some(reexport) = reexport {
                    *name = reexport;
                }
            }
            // Re-exported `extern crate` (#43189).
            DefPathData::CrateRoot => {
                data = DefPathData::TypeNs(self.tcx().original_crate_name(def_id.krate));
            }
            _ => {}
        }
        debug!("try_print_visible_def_path: data={:?}", data);

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
                if self.region_should_not_be_omitted(r) {
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
                let sig = self.tcx().fn_sig(def_id).subst(self.tcx(), substs);
                p!(print(sig), " {{", print_value_path(def_id, substs), "}}");
            }
            ty::FnPtr(ref bare_fn) => p!(print(bare_fn)),
            ty::Infer(infer_ty) => {
                if let ty::TyVar(ty_vid) = infer_ty {
                    if let Some(name) = self.infer_ty_name(ty_vid) {
                        p!(write("{}", name))
                    } else {
                        p!(write("{}", infer_ty))
                    }
                } else {
                    p!(write("{}", infer_ty))
                }
            }
            ty::Error(_) => p!("[type error]"),
            ty::Param(ref param_ty) => p!(write("{}", param_ty)),
            ty::Bound(debruijn, bound_ty) => match bound_ty.kind {
                ty::BoundTyKind::Anon => self.pretty_print_bound_var(debruijn, bound_ty.var)?,
                ty::BoundTyKind::Param(p) => p!(write("{}", p)),
            },
            ty::Adt(def, substs) => {
                p!(print_def_path(def.did, substs));
            }
            ty::Dynamic(data, r) => {
                let print_r = self.region_should_not_be_omitted(r);
                if print_r {
                    p!("(");
                }
                p!("dyn ", print(data));
                if print_r {
                    p!(" + ", print(r), ")");
                }
            }
            ty::Foreign(def_id) => {
                p!(print_def_path(def_id, &[]));
            }
            ty::Projection(ref data) => p!(print(data)),
            ty::Placeholder(placeholder) => p!(write("Placeholder({:?})", placeholder)),
            ty::Opaque(def_id, substs) => {
                // FIXME(eddyb) print this with `print_def_path`.
                // We use verbose printing in 'NO_QUERIES' mode, to
                // avoid needing to call `predicates_of`. This should
                // only affect certain debug messages (e.g. messages printed
                // from `rustc_middle::ty` during the computation of `tcx.predicates_of`),
                // and should have no effect on any compiler output.
                if self.tcx().sess.verbose() || NO_QUERIES.with(|q| q.get()) {
                    p!(write("Opaque({:?}, {:?})", def_id, substs));
                    return Ok(self);
                }

                return Ok(with_no_queries(|| {
                    let def_key = self.tcx().def_key(def_id);
                    if let Some(name) = def_key.disambiguated_data.data.get_opt_name() {
                        p!(write("{}", name));
                        // FIXME(eddyb) print this with `print_def_path`.
                        if !substs.is_empty() {
                            p!("::");
                            p!(generic_delimiters(|cx| cx.comma_sep(substs.iter())));
                        }
                        return Ok(self);
                    }
                    // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                    // by looking up the projections associated with the def_id.
                    let bounds = self.tcx().explicit_item_bounds(def_id);

                    let mut first = true;
                    let mut is_sized = false;
                    p!("impl");
                    for (predicate, _) in bounds {
                        let predicate = predicate.subst(self.tcx(), substs);
                        // Note: We can't use `to_opt_poly_trait_ref` here as `predicate`
                        // may contain unbound variables. We therefore do this manually.
                        //
                        // FIXME(lcnr): Find out why exactly this is the case :)
                        let bound_predicate = predicate.bound_atom_with_opt_escaping(self.tcx());
                        if let ty::PredicateAtom::Trait(pred, _) = bound_predicate.skip_binder() {
                            let trait_ref = bound_predicate.rebind(pred.trait_ref);
                            // Don't print +Sized, but rather +?Sized if absent.
                            if Some(trait_ref.def_id()) == self.tcx().lang_items().sized_trait() {
                                is_sized = true;
                                continue;
                            }

                            p!(
                                write("{}", if first { " " } else { "+" }),
                                print(trait_ref.print_only_trait_path())
                            );
                            first = false;
                        }
                    }
                    if !is_sized {
                        p!(write("{}?Sized", if first { " " } else { "+" }));
                    } else if first {
                        p!(" Sized");
                    }
                    Ok(self)
                })?);
            }
            ty::Str => p!("str"),
            ty::Generator(did, substs, movability) => {
                p!(write("["));
                match movability {
                    hir::Movability::Movable => {}
                    hir::Movability::Static => p!("static "),
                }

                if !self.tcx().sess.verbose() {
                    p!("generator");
                    // FIXME(eddyb) should use `def_span`.
                    if let Some(did) = did.as_local() {
                        let hir_id = self.tcx().hir().local_def_id_to_hir_id(did);
                        let span = self.tcx().hir().span(hir_id);
                        p!(write("@{}", self.tcx().sess.source_map().span_to_string(span)));
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
                }

                if substs.as_generator().is_valid() {
                    p!(" ", print(substs.as_generator().witness()));
                }

                p!("]")
            }
            ty::GeneratorWitness(types) => {
                p!(in_binder(&types));
            }
            ty::Closure(did, substs) => {
                p!(write("["));
                if !self.tcx().sess.verbose() {
                    p!(write("closure"));
                    // FIXME(eddyb) should use `def_span`.
                    if let Some(did) = did.as_local() {
                        let hir_id = self.tcx().hir().local_def_id_to_hir_id(did);
                        if self.tcx().sess.opts.debugging_opts.span_free_formats {
                            p!("@", print_def_path(did.to_def_id(), substs));
                        } else {
                            let span = self.tcx().hir().span(hir_id);
                            p!(write("@{}", self.tcx().sess.source_map().span_to_string(span)));
                        }
                    } else {
                        p!(write("@"), print_def_path(did, substs));
                    }
                } else {
                    p!(print_def_path(did, substs));
                    if !substs.as_closure().is_valid() {
                        p!(" closure_substs=(unavailable)");
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
            ty::Array(ty, sz) => {
                p!("[", print(ty), "; ");
                if self.tcx().sess.verbose() {
                    p!(write("{:?}", sz));
                } else if let ty::ConstKind::Unevaluated(..) = sz.val {
                    // Do not try to evaluate unevaluated constants. If we are const evaluating an
                    // array length anon const, rustc will (with debug assertions) print the
                    // constant's path. Which will end up here again.
                    p!("_");
                } else if let Some(n) = sz.val.try_to_bits(self.tcx().data_layout.pointer_size) {
                    p!(write("{}", n));
                } else if let ty::ConstKind::Param(param) = sz.val {
                    p!(write("{}", param));
                } else {
                    p!("_");
                }
                p!("]")
            }
            ty::Slice(ty) => p!("[", print(ty), "]"),
        }

        Ok(self)
    }

    fn pretty_print_bound_var(
        &mut self,
        debruijn: ty::DebruijnIndex,
        var: ty::BoundVar,
    ) -> Result<(), Self::Error> {
        if debruijn == ty::INNERMOST {
            write!(self, "^{}", var.index())
        } else {
            write!(self, "^{}_{}", debruijn.index(), var.index())
        }
    }

    fn infer_ty_name(&self, _: ty::TyVid) -> Option<String> {
        None
    }

    fn pretty_print_dyn_existential(
        mut self,
        predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        define_scoped_cx!(self);

        // Generate the main trait ref, including associated types.
        let mut first = true;

        if let Some(principal) = predicates.principal() {
            p!(print_def_path(principal.def_id, &[]));

            let mut resugared = false;

            // Special-case `Fn(...) -> ...` and resugar it.
            let fn_trait_kind = self.tcx().fn_trait_kind_from_lang_item(principal.def_id);
            if !self.tcx().sess.verbose() && fn_trait_kind.is_some() {
                if let ty::Tuple(ref args) = principal.substs.type_at(0).kind() {
                    let mut projections = predicates.projection_bounds();
                    if let (Some(proj), None) = (projections.next(), projections.next()) {
                        let tys: Vec<_> = args.iter().map(|k| k.expect_ty()).collect();
                        p!(pretty_fn_sig(&tys, false, proj.ty));
                        resugared = true;
                    }
                }
            }

            // HACK(eddyb) this duplicates `FmtPrinter`'s `path_generic_args`,
            // in order to place the projections inside the `<...>`.
            if !resugared {
                // Use a type that can't appear in defaults of type parameters.
                let dummy_self = self.tcx().mk_ty_infer(ty::FreshTy(0));
                let principal = principal.with_self_ty(self.tcx(), dummy_self);

                let args = self.generic_args_to_print(
                    self.tcx().generics_of(principal.def_id),
                    principal.substs,
                );

                // Don't print `'_` if there's no unerased regions.
                let print_regions = args.iter().any(|arg| match arg.unpack() {
                    GenericArgKind::Lifetime(r) => *r != ty::ReErased,
                    _ => false,
                });
                let mut args = args.iter().cloned().filter(|arg| match arg.unpack() {
                    GenericArgKind::Lifetime(_) => print_regions,
                    _ => true,
                });
                let mut projections = predicates.projection_bounds();

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
            first = false;
        }

        // Builtin bounds.
        // FIXME(eddyb) avoid printing twice (needed to ensure
        // that the auto traits are sorted *and* printed via cx).
        let mut auto_traits: Vec<_> =
            predicates.auto_traits().map(|did| (self.tcx().def_path_str(did), did)).collect();

        // The auto traits come ordered by `DefPathHash`. While
        // `DefPathHash` is *stable* in the sense that it depends on
        // neither the host nor the phase of the moon, it depends
        // "pseudorandomly" on the compiler version and the target.
        //
        // To avoid that causing instabilities in compiletest
        // output, sort the auto-traits alphabetically.
        auto_traits.sort();

        for (_, def_id) in auto_traits {
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
        ct: &'tcx ty::Const<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        if self.tcx().sess.verbose() {
            p!(write("Const({:?}: {:?})", ct.val, ct.ty));
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
                        |this| this.print_type(ct.ty),
                        ": ",
                    )?;
                } else {
                    write!(self, "_")?;
                }
            }};
        }

        match ct.val {
            ty::ConstKind::Unevaluated(def, substs, promoted) => {
                if let Some(promoted) = promoted {
                    p!(print_value_path(def.did, substs));
                    p!(write("::{:?}", promoted));
                } else {
                    match self.tcx().def_kind(def.did) {
                        DefKind::Static | DefKind::Const | DefKind::AssocConst => {
                            p!(print_value_path(def.did, substs))
                        }
                        _ => {
                            if def.is_local() {
                                let span = self.tcx().def_span(def.did);
                                if let Ok(snip) = self.tcx().sess.source_map().span_to_snippet(span)
                                {
                                    p!(write("{}", snip))
                                } else {
                                    print_underscore!()
                                }
                            } else {
                                print_underscore!()
                            }
                        }
                    }
                }
            }
            ty::ConstKind::Infer(..) => print_underscore!(),
            ty::ConstKind::Param(ParamConst { name, .. }) => p!(write("{}", name)),
            ty::ConstKind::Value(value) => {
                return self.pretty_print_const_value(value, ct.ty, print_ty);
            }

            ty::ConstKind::Bound(debruijn, bound_var) => {
                self.pretty_print_bound_var(debruijn, bound_var)?
            }
            ty::ConstKind::Placeholder(placeholder) => p!(write("Placeholder({:?})", placeholder)),
            ty::ConstKind::Error(_) => p!("[const error]"),
        };
        Ok(self)
    }

    fn pretty_print_const_scalar(
        mut self,
        scalar: Scalar,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        match (scalar, &ty.kind()) {
            // Byte strings (&[u8; N])
            (
                Scalar::Ptr(ptr),
                ty::Ref(
                    _,
                    ty::TyS {
                        kind:
                            ty::Array(
                                ty::TyS { kind: ty::Uint(ast::UintTy::U8), .. },
                                ty::Const {
                                    val:
                                        ty::ConstKind::Value(ConstValue::Scalar(Scalar::Raw {
                                            data,
                                            ..
                                        })),
                                    ..
                                },
                            ),
                        ..
                    },
                    _,
                ),
            ) => match self.tcx().get_global_alloc(ptr.alloc_id) {
                Some(GlobalAlloc::Memory(alloc)) => {
                    if let Ok(byte_str) = alloc.get_bytes(&self.tcx(), ptr, Size::from_bytes(*data))
                    {
                        p!(pretty_print_byte_str(byte_str))
                    } else {
                        p!("<too short allocation>")
                    }
                }
                // FIXME: for statics and functions, we could in principle print more detail.
                Some(GlobalAlloc::Static(def_id)) => p!(write("<static({:?})>", def_id)),
                Some(GlobalAlloc::Function(_)) => p!("<function>"),
                None => p!("<dangling pointer>"),
            },
            // Bool
            (Scalar::Raw { data: 0, .. }, ty::Bool) => p!("false"),
            (Scalar::Raw { data: 1, .. }, ty::Bool) => p!("true"),
            // Float
            (Scalar::Raw { data, .. }, ty::Float(ast::FloatTy::F32)) => {
                p!(write("{}f32", Single::from_bits(data)))
            }
            (Scalar::Raw { data, .. }, ty::Float(ast::FloatTy::F64)) => {
                p!(write("{}f64", Double::from_bits(data)))
            }
            // Int
            (Scalar::Raw { data, .. }, ty::Uint(ui)) => {
                let size = Integer::from_attr(&self.tcx(), UnsignedInt(*ui)).size();
                let int = ConstInt::new(data, size, false, ty.is_ptr_sized_integral());
                if print_ty { p!(write("{:#?}", int)) } else { p!(write("{:?}", int)) }
            }
            (Scalar::Raw { data, .. }, ty::Int(i)) => {
                let size = Integer::from_attr(&self.tcx(), SignedInt(*i)).size();
                let int = ConstInt::new(data, size, true, ty.is_ptr_sized_integral());
                if print_ty { p!(write("{:#?}", int)) } else { p!(write("{:?}", int)) }
            }
            // Char
            (Scalar::Raw { data, .. }, ty::Char) if char::from_u32(data as u32).is_some() => {
                p!(write("{:?}", char::from_u32(data as u32).unwrap()))
            }
            // Raw pointers
            (Scalar::Raw { data, .. }, ty::RawPtr(_)) => {
                self = self.typed_value(
                    |mut this| {
                        write!(this, "0x{:x}", data)?;
                        Ok(this)
                    },
                    |this| this.print_type(ty),
                    " as ",
                )?;
            }
            (Scalar::Ptr(ptr), ty::FnPtr(_)) => {
                // FIXME: this can ICE when the ptr is dangling or points to a non-function.
                // We should probably have a helper method to share code with the "Byte strings"
                // printing above (which also has to handle pointers to all sorts of things).
                let instance = self.tcx().global_alloc(ptr.alloc_id).unwrap_fn();
                self = self.typed_value(
                    |this| this.print_value_path(instance.def_id(), instance.substs),
                    |this| this.print_type(ty),
                    " as ",
                )?;
            }
            // For function type zsts just printing the path is enough
            (Scalar::Raw { size: 0, .. }, ty::FnDef(d, s)) => p!(print_value_path(*d, s)),
            // Nontrivial types with scalar bit representation
            (Scalar::Raw { data, size }, _) => {
                let print = |mut this: Self| {
                    if size == 0 {
                        write!(this, "transmute(())")?;
                    } else {
                        write!(this, "transmute(0x{:01$x})", data, size as usize * 2)?;
                    }
                    Ok(this)
                };
                self = if print_ty {
                    self.typed_value(print, |this| this.print_type(ty), ": ")?
                } else {
                    print(self)?
                };
            }
            // Any pointer values not covered by a branch above
            (Scalar::Ptr(p), _) => {
                self = self.pretty_print_const_pointer(p, ty, print_ty)?;
            }
        }
        Ok(self)
    }

    /// This is overridden for MIR printing because we only want to hide alloc ids from users, not
    /// from MIR where it is actually useful.
    fn pretty_print_const_pointer(
        mut self,
        _: Pointer,
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
        define_scoped_cx!(self);
        p!("b\"");
        for &c in byte_str {
            for e in std::ascii::escape_default(c) {
                self.write_char(e as char)?;
            }
        }
        p!("\"");
        Ok(self)
    }

    fn pretty_print_const_value(
        mut self,
        ct: ConstValue<'tcx>,
        ty: Ty<'tcx>,
        print_ty: bool,
    ) -> Result<Self::Const, Self::Error> {
        define_scoped_cx!(self);

        if self.tcx().sess.verbose() {
            p!(write("ConstValue({:?}: ", ct), print(ty), ")");
            return Ok(self);
        }

        let u8_type = self.tcx().types.u8;

        match (ct, ty.kind()) {
            // Byte/string slices, printed as (byte) string literals.
            (
                ConstValue::Slice { data, start, end },
                ty::Ref(_, ty::TyS { kind: ty::Slice(t), .. }, _),
            ) if *t == u8_type => {
                // The `inspect` here is okay since we checked the bounds, and there are
                // no relocations (we have an active slice reference here). We don't use
                // this result to affect interpreter execution.
                let byte_str = data.inspect_with_uninit_and_ptr_outside_interpreter(start..end);
                self.pretty_print_byte_str(byte_str)
            }
            (
                ConstValue::Slice { data, start, end },
                ty::Ref(_, ty::TyS { kind: ty::Str, .. }, _),
            ) => {
                // The `inspect` here is okay since we checked the bounds, and there are no
                // relocations (we have an active `str` reference here). We don't use this
                // result to affect interpreter execution.
                let slice = data.inspect_with_uninit_and_ptr_outside_interpreter(start..end);
                let s = std::str::from_utf8(slice).expect("non utf8 str from miri");
                p!(write("{:?}", s));
                Ok(self)
            }
            (ConstValue::ByRef { alloc, offset }, ty::Array(t, n)) if *t == u8_type => {
                let n = n.val.try_to_bits(self.tcx().data_layout.pointer_size).unwrap();
                // cast is ok because we already checked for pointer size (32 or 64 bit) above
                let n = Size::from_bytes(n);
                let ptr = Pointer::new(AllocId(0), offset);

                let byte_str = alloc.get_bytes(&self.tcx(), ptr, n).unwrap();
                p!("*");
                p!(pretty_print_byte_str(byte_str));
                Ok(self)
            }

            // Aggregates, printed as array/tuple/struct/variant construction syntax.
            //
            // NB: the `has_param_types_or_consts` check ensures that we can use
            // the `destructure_const` query with an empty `ty::ParamEnv` without
            // introducing ICEs (e.g. via `layout_of`) from missing bounds.
            // E.g. `transmute([0usize; 2]): (u8, *mut T)` needs to know `T: Sized`
            // to be able to destructure the tuple into `(0u8, *mut T)
            //
            // FIXME(eddyb) for `--emit=mir`/`-Z dump-mir`, we should provide the
            // correct `ty::ParamEnv` to allow printing *all* constant values.
            (_, ty::Array(..) | ty::Tuple(..) | ty::Adt(..)) if !ty.has_param_types_or_consts() => {
                let contents = self.tcx().destructure_const(
                    ty::ParamEnv::reveal_all()
                        .and(self.tcx().mk_const(ty::Const { val: ty::ConstKind::Value(ct), ty })),
                );
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
                    ty::Adt(def, substs) if def.variants.is_empty() => {
                        p!(print_value_path(def.did, substs));
                    }
                    ty::Adt(def, substs) => {
                        let variant_id =
                            contents.variant.expect("destructed const of adt without variant id");
                        let variant_def = &def.variants[variant_id];
                        p!(print_value_path(variant_def.def_id, substs));

                        match variant_def.ctor_kind {
                            CtorKind::Const => {}
                            CtorKind::Fn => {
                                p!("(", comma_sep(fields), ")");
                            }
                            CtorKind::Fictive => {
                                p!(" {{ ");
                                let mut first = true;
                                for (field_def, field) in variant_def.fields.iter().zip(fields) {
                                    if !first {
                                        p!(", ");
                                    }
                                    p!(write("{}: ", field_def.ident), print(field));
                                    first = false;
                                }
                                p!(" }}");
                            }
                        }
                    }
                    _ => unreachable!(),
                }

                Ok(self)
            }

            (ConstValue::Scalar(scalar), _) => self.pretty_print_const_scalar(scalar, ty, print_ty),

            // FIXME(oli-obk): also pretty print arrays and other aggregate constants by reading
            // their fields instead of just dumping the memory.
            _ => {
                // fallback
                p!(write("{:?}", ct));
                if print_ty {
                    p!(": ", print(ty));
                }
                Ok(self)
            }
        }
    }
}

// HACK(eddyb) boxed to avoid moving around a large struct by-value.
pub struct FmtPrinter<'a, 'tcx, F>(Box<FmtPrinterData<'a, 'tcx, F>>);

pub struct FmtPrinterData<'a, 'tcx, F> {
    tcx: TyCtxt<'tcx>,
    fmt: F,

    empty_path: bool,
    in_value: bool,
    pub print_alloc_ids: bool,

    used_region_names: FxHashSet<Symbol>,
    region_index: usize,
    binder_depth: usize,
    printed_type_count: usize,

    pub region_highlight_mode: RegionHighlightMode,

    pub name_resolver: Option<Box<&'a dyn Fn(ty::sty::TyVid) -> Option<String>>>,
}

impl<F> Deref for FmtPrinter<'a, 'tcx, F> {
    type Target = FmtPrinterData<'a, 'tcx, F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for FmtPrinter<'_, '_, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F> FmtPrinter<'a, 'tcx, F> {
    pub fn new(tcx: TyCtxt<'tcx>, fmt: F, ns: Namespace) -> Self {
        FmtPrinter(Box::new(FmtPrinterData {
            tcx,
            fmt,
            empty_path: false,
            in_value: ns == Namespace::ValueNS,
            print_alloc_ids: false,
            used_region_names: Default::default(),
            region_index: 0,
            binder_depth: 0,
            printed_type_count: 0,
            region_highlight_mode: RegionHighlightMode::default(),
            name_resolver: None,
        }))
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

impl TyCtxt<'t> {
    /// Returns a string identifying this `DefId`. This string is
    /// suitable for user output.
    pub fn def_path_str(self, def_id: DefId) -> String {
        self.def_path_str_with_substs(def_id, &[])
    }

    pub fn def_path_str_with_substs(self, def_id: DefId, substs: &'t [GenericArg<'t>]) -> String {
        let ns = guess_def_namespace(self, def_id);
        debug!("def_path_str: def_id={:?}, ns={:?}", def_id, ns);
        let mut s = String::new();
        let _ = FmtPrinter::new(self, &mut s, ns).print_def_path(def_id, substs);
        s
    }
}

impl<F: fmt::Write> fmt::Write for FmtPrinter<'_, '_, F> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.fmt.write_str(s)
    }
}

impl<F: fmt::Write> Printer<'tcx> for FmtPrinter<'_, 'tcx, F> {
    type Error = fmt::Error;

    type Path = Self;
    type Region = Self;
    type Type = Self;
    type DynExistential = Self;
    type Const = Self;

    fn tcx(&'a self) -> TyCtxt<'tcx> {
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
                write!(self, "<impl at {}>", self.tcx.sess.source_map().span_to_string(span))?;
                self.empty_path = false;

                return Ok(self);
            }
        }

        self.default_print_def_path(def_id, substs)
    }

    fn print_region(self, region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
        self.pretty_print_region(region)
    }

    fn print_type(mut self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
        if self.tcx.sess.type_length_limit().value_within_limit(self.printed_type_count) {
            self.printed_type_count += 1;
            self.pretty_print_type(ty)
        } else {
            write!(self, "...")?;
            Ok(self)
        }
    }

    fn print_dyn_existential(
        self,
        predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        self.pretty_print_dyn_existential(predicates)
    }

    fn print_const(self, ct: &'tcx ty::Const<'tcx>) -> Result<Self::Const, Self::Error> {
        self.pretty_print_const(ct, true)
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

        // Skip `::{{constructor}}` on tuple/unit structs.
        if let DefPathData::Ctor = disambiguated_data.data {
            return Ok(self);
        }

        // FIXME(eddyb) `name` should never be empty, but it
        // currently is for `extern { ... }` "foreign modules".
        let name = disambiguated_data.data.name();
        if name != DefPathDataName::Named(kw::Invalid) {
            if !self.empty_path {
                write!(self, "::")?;
            }

            if let DefPathDataName::Named(name) = name {
                if Ident::with_dummy_span(name).is_raw_guess() {
                    write!(self, "r#")?;
                }
            }

            let verbose = self.tcx.sess.verbose();
            disambiguated_data.fmt_maybe_verbose(&mut self, verbose)?;

            self.empty_path = false;
        }

        Ok(self)
    }

    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        // Don't print `'_` if there's no unerased regions.
        let print_regions = args.iter().any(|arg| match arg.unpack() {
            GenericArgKind::Lifetime(r) => *r != ty::ReErased,
            _ => false,
        });
        let args = args.iter().cloned().filter(|arg| match arg.unpack() {
            GenericArgKind::Lifetime(_) => print_regions,
            _ => true,
        });

        if args.clone().next().is_some() {
            if self.in_value {
                write!(self, "::")?;
            }
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(self)
        }
    }
}

impl<F: fmt::Write> PrettyPrinter<'tcx> for FmtPrinter<'_, 'tcx, F> {
    fn infer_ty_name(&self, id: ty::TyVid) -> Option<String> {
        self.0.name_resolver.as_ref().and_then(|func| func(id))
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

    fn in_binder<T>(self, value: &ty::Binder<T>) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error> + TypeFoldable<'tcx>,
    {
        self.pretty_in_binder(value)
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

    fn region_should_not_be_omitted(&self, region: ty::Region<'_>) -> bool {
        let highlight = self.region_highlight_mode;
        if highlight.region_highlighted(region).is_some() {
            return true;
        }

        if self.tcx.sess.verbose() {
            return true;
        }

        let identify_regions = self.tcx.sess.opts.debugging_opts.identify_regions;

        match *region {
            ty::ReEarlyBound(ref data) => {
                data.name != kw::Invalid && data.name != kw::UnderscoreLifetime
            }

            ty::ReLateBound(_, br)
            | ty::ReFree(ty::FreeRegion { bound_region: br, .. })
            | ty::RePlaceholder(ty::Placeholder { name: br, .. }) => {
                if let ty::BrNamed(_, name) = br {
                    if name != kw::Invalid && name != kw::UnderscoreLifetime {
                        return true;
                    }
                }

                if let Some((region, _)) = highlight.highlight_bound_region {
                    if br == region {
                        return true;
                    }
                }

                false
            }

            ty::ReVar(_) if identify_regions => true,

            ty::ReVar(_) | ty::ReErased => false,

            ty::ReStatic | ty::ReEmpty(_) => true,
        }
    }

    fn pretty_print_const_pointer(
        self,
        p: Pointer,
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
impl<F: fmt::Write> FmtPrinter<'_, '_, F> {
    pub fn pretty_print_region(mut self, region: ty::Region<'_>) -> Result<Self, fmt::Error> {
        define_scoped_cx!(self);

        // Watch out for region highlights.
        let highlight = self.region_highlight_mode;
        if let Some(n) = highlight.region_highlighted(region) {
            p!(write("'{}", n));
            return Ok(self);
        }

        if self.tcx.sess.verbose() {
            p!(write("{:?}", region));
            return Ok(self);
        }

        let identify_regions = self.tcx.sess.opts.debugging_opts.identify_regions;

        // These printouts are concise.  They do not contain all the information
        // the user might want to diagnose an error, but there is basically no way
        // to fit that into a short string.  Hence the recommendation to use
        // `explain_region()` or `note_and_explain_region()`.
        match *region {
            ty::ReEarlyBound(ref data) => {
                if data.name != kw::Invalid {
                    p!(write("{}", data.name));
                    return Ok(self);
                }
            }
            ty::ReLateBound(_, br)
            | ty::ReFree(ty::FreeRegion { bound_region: br, .. })
            | ty::RePlaceholder(ty::Placeholder { name: br, .. }) => {
                if let ty::BrNamed(_, name) = br {
                    if name != kw::Invalid && name != kw::UnderscoreLifetime {
                        p!(write("{}", name));
                        return Ok(self);
                    }
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
            ty::ReStatic => {
                p!("'static");
                return Ok(self);
            }
            ty::ReEmpty(ty::UniverseIndex::ROOT) => {
                p!("'<empty>");
                return Ok(self);
            }
            ty::ReEmpty(ui) => {
                p!(write("'<empty:{:?}>", ui));
                return Ok(self);
            }
        }

        p!("'_");

        Ok(self)
    }
}

// HACK(eddyb) limited to `FmtPrinter` because of `binder_depth`,
// `region_index` and `used_region_names`.
impl<F: fmt::Write> FmtPrinter<'_, 'tcx, F> {
    pub fn name_all_regions<T>(
        mut self,
        value: &ty::Binder<T>,
    ) -> Result<(Self, (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)), fmt::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = fmt::Error> + TypeFoldable<'tcx>,
    {
        fn name_by_region_index(index: usize) -> Symbol {
            match index {
                0 => Symbol::intern("'r"),
                1 => Symbol::intern("'s"),
                i => Symbol::intern(&format!("'t{}", i - 2)),
            }
        }

        // Replace any anonymous late-bound regions with named
        // variants, using new unique identifiers, so that we can
        // clearly differentiate between named and unnamed regions in
        // the output. We'll probably want to tweak this over time to
        // decide just how much information to give.
        if self.binder_depth == 0 {
            self.prepare_late_bound_region_info(value);
        }

        let mut empty = true;
        let mut start_or_continue = |cx: &mut Self, start: &str, cont: &str| {
            write!(
                cx,
                "{}",
                if empty {
                    empty = false;
                    start
                } else {
                    cont
                }
            )
        };

        define_scoped_cx!(self);

        let mut region_index = self.region_index;
        let new_value = self.tcx.replace_late_bound_regions(value, |br| {
            let _ = start_or_continue(&mut self, "for<", ", ");
            let br = match br {
                ty::BrNamed(_, name) => {
                    let _ = write!(self, "{}", name);
                    br
                }
                ty::BrAnon(_) | ty::BrEnv => {
                    let name = loop {
                        let name = name_by_region_index(region_index);
                        region_index += 1;
                        if !self.used_region_names.contains(&name) {
                            break name;
                        }
                    };
                    let _ = write!(self, "{}", name);
                    ty::BrNamed(DefId::local(CRATE_DEF_INDEX), name)
                }
            };
            self.tcx.mk_region(ty::ReLateBound(ty::INNERMOST, br))
        });
        start_or_continue(&mut self, "", "> ")?;

        self.binder_depth += 1;
        self.region_index = region_index;
        Ok((self, new_value))
    }

    pub fn pretty_in_binder<T>(self, value: &ty::Binder<T>) -> Result<Self, fmt::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = fmt::Error> + TypeFoldable<'tcx>,
    {
        let old_region_index = self.region_index;
        let (new, new_value) = self.name_all_regions(value)?;
        let mut inner = new_value.0.print(new)?;
        inner.region_index = old_region_index;
        inner.binder_depth -= 1;
        Ok(inner)
    }

    fn prepare_late_bound_region_info<T>(&mut self, value: &ty::Binder<T>)
    where
        T: TypeFoldable<'tcx>,
    {
        struct LateBoundRegionNameCollector<'a>(&'a mut FxHashSet<Symbol>);
        impl<'tcx> ty::fold::TypeVisitor<'tcx> for LateBoundRegionNameCollector<'_> {
            fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<()> {
                if let ty::ReLateBound(_, ty::BrNamed(_, name)) = *r {
                    self.0.insert(name);
                }
                r.super_visit_with(self)
            }
        }

        self.used_region_names.clear();
        let mut collector = LateBoundRegionNameCollector(&mut self.used_region_names);
        value.visit_with(&mut collector);
        self.region_index = 0;
    }
}

impl<'tcx, T, P: PrettyPrinter<'tcx>> Print<'tcx, P> for ty::Binder<T>
where
    T: Print<'tcx, P, Output = P, Error = P::Error> + TypeFoldable<'tcx>,
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
        $(impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                ty::tls::with(|tcx| {
                    tcx.lift(*self)
                        .expect("could not lift for printing")
                        .print(FmtPrinter::new(tcx, f, Namespace::TypeNS))?;
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

// HACK(eddyb) this is separate because `ty::RegionKind` doesn't need lifting.
impl fmt::Display for ty::RegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            self.print(FmtPrinter::new(tcx, f, Namespace::TypeNS))?;
            Ok(())
        })
    }
}

/// Wrapper type for `ty::TraitRef` which opts-in to pretty printing only
/// the trait path. That is, it will print `Trait<U>` instead of
/// `<T as Trait<U>>`.
#[derive(Copy, Clone, TypeFoldable, Lift)]
pub struct TraitRefPrintOnlyTraitPath<'tcx>(ty::TraitRef<'tcx>);

impl fmt::Debug for TraitRefPrintOnlyTraitPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl ty::TraitRef<'tcx> {
    pub fn print_only_trait_path(self) -> TraitRefPrintOnlyTraitPath<'tcx> {
        TraitRefPrintOnlyTraitPath(self)
    }
}

impl ty::Binder<ty::TraitRef<'tcx>> {
    pub fn print_only_trait_path(self) -> ty::Binder<TraitRefPrintOnlyTraitPath<'tcx>> {
        self.map_bound(|tr| tr.print_only_trait_path())
    }
}

forward_display_to_print! {
    Ty<'tcx>,
    &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    &'tcx ty::Const<'tcx>,

    // HACK(eddyb) these are exhaustive instead of generic,
    // because `for<'tcx>` isn't possible yet.
    ty::Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>,
    ty::Binder<ty::TraitRef<'tcx>>,
    ty::Binder<TraitRefPrintOnlyTraitPath<'tcx>>,
    ty::Binder<ty::FnSig<'tcx>>,
    ty::Binder<ty::TraitPredicate<'tcx>>,
    ty::Binder<ty::SubtypePredicate<'tcx>>,
    ty::Binder<ty::ProjectionPredicate<'tcx>>,
    ty::Binder<ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>,
    ty::Binder<ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>>,

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
        let dummy_self = cx.tcx().mk_ty_infer(ty::FreshTy(0));
        let trait_ref = self.with_self_ty(cx.tcx(), dummy_self);
        p!(print(trait_ref.print_only_trait_path()))
    }

    ty::ExistentialProjection<'tcx> {
        let name = cx.tcx().associated_item(self.item_def_id).ident;
        p!(write("{} = ", name), print(self.ty))
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

    ty::InferTy {
        if cx.tcx().sess.verbose() {
            p!(write("{:?}", self));
            return Ok(cx);
        }
        match *self {
            ty::TyVar(_) => p!("_"),
            ty::IntVar(_) => p!(write("{}", "{integer}")),
            ty::FloatVar(_) => p!(write("{}", "{float}")),
            ty::FreshTy(v) => p!(write("FreshTy({})", v)),
            ty::FreshIntTy(v) => p!(write("FreshIntTy({})", v)),
            ty::FreshFloatTy(v) => p!(write("FreshFloatTy({})", v))
        }
    }

    ty::TraitRef<'tcx> {
        p!(write("<{} as {}>", self.self_ty(), self.print_only_trait_path()))
    }

    TraitRefPrintOnlyTraitPath<'tcx> {
        p!(print_def_path(self.0.def_id, self.0.substs));
    }

    ty::ParamTy {
        p!(write("{}", self.name))
    }

    ty::ParamConst {
        p!(write("{}", self.name))
    }

    ty::SubtypePredicate<'tcx> {
        p!(print(self.a), " <: ", print(self.b))
    }

    ty::TraitPredicate<'tcx> {
        p!(print(self.trait_ref.self_ty()), ": ",
           print(self.trait_ref.print_only_trait_path()))
    }

    ty::ProjectionPredicate<'tcx> {
        p!(print(self.projection_ty), " == ", print(self.ty))
    }

    ty::ProjectionTy<'tcx> {
        p!(print_def_path(self.item_def_id, self.substs));
    }

    ty::ClosureKind {
        match *self {
            ty::ClosureKind::Fn => p!("Fn"),
            ty::ClosureKind::FnMut => p!("FnMut"),
            ty::ClosureKind::FnOnce => p!("FnOnce"),
        }
    }

    ty::Predicate<'tcx> {
        match self.kind() {
            &ty::PredicateKind::Atom(atom) => p!(print(atom)),
            ty::PredicateKind::ForAll(binder) => p!(print(binder)),
        }
    }

    ty::PredicateAtom<'tcx> {
        match *self {
            ty::PredicateAtom::Trait(ref data, constness) => {
                if let hir::Constness::Const = constness {
                    p!("const ");
                }
                p!(print(data))
            }
            ty::PredicateAtom::Subtype(predicate) => p!(print(predicate)),
            ty::PredicateAtom::RegionOutlives(predicate) => p!(print(predicate)),
            ty::PredicateAtom::TypeOutlives(predicate) => p!(print(predicate)),
            ty::PredicateAtom::Projection(predicate) => p!(print(predicate)),
            ty::PredicateAtom::WellFormed(arg) => p!(print(arg), " well-formed"),
            ty::PredicateAtom::ObjectSafe(trait_def_id) => {
                p!("the trait `", print_def_path(trait_def_id, &[]), "` is object-safe")
            }
            ty::PredicateAtom::ClosureKind(closure_def_id, _closure_substs, kind) => {
                p!("the closure `",
                print_value_path(closure_def_id, &[]),
                write("` implements the trait `{}`", kind))
            }
            ty::PredicateAtom::ConstEvaluatable(def, substs) => {
                p!("the constant `", print_value_path(def.did, substs), "` can be evaluated")
            }
            ty::PredicateAtom::ConstEquate(c1, c2) => {
                p!("the constant `", print(c1), "` equals `", print(c2), "`")
            }
            ty::PredicateAtom::TypeWellFormedFromEnv(ty) => {
                p!("the type `", print(ty), "` is found in the environment")
            }
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
    for item in hir.krate().items.values() {
        if item.ident.name.as_str().is_empty() || matches!(item.kind, ItemKind::Use(_, _)) {
            continue;
        }

        if let Some(local_def_id) = hir.definitions().opt_hir_id_to_local_def_id(item.hir_id) {
            let def_id = local_def_id.to_def_id();
            let ns = tcx.def_kind(def_id).ns().unwrap_or(Namespace::TypeNS);
            collect_fn(&item.ident, ns, def_id);
        }
    }

    // Now take care of extern crate items.
    let queue = &mut Vec::new();
    let mut seen_defs: DefIdSet = Default::default();

    for &cnum in tcx.crates().iter() {
        let def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };

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
        for child in tcx.item_children(def).iter() {
            if child.vis != ty::Visibility::Public {
                continue;
            }

            match child.res {
                def::Res::Def(DefKind::AssocTy, _) => {}
                def::Res::Def(defkind, def_id) => {
                    if let Some(ns) = defkind.ns() {
                        collect_fn(&child.ident, ns, def_id);
                    }

                    if seen_defs.insert(def_id) {
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
fn trimmed_def_paths(tcx: TyCtxt<'_>, crate_num: CrateNum) -> FxHashMap<DefId, Symbol> {
    assert_eq!(crate_num, LOCAL_CRATE);

    let mut map = FxHashMap::default();

    if let TrimmedDefPaths::GoodPath = tcx.sess.opts.trimmed_def_paths {
        // For good paths causing this bug, the `rustc_middle::ty::print::with_no_trimmed_paths`
        // wrapper can be used to suppress this query, in exchange for full paths being formatted.
        tcx.sess.delay_good_path_bug("trimmed_def_paths constructed");
    }

    let unique_symbols_rev: &mut FxHashMap<(Namespace, Symbol), Option<DefId>> =
        &mut FxHashMap::default();

    for symbol_set in tcx.glob_map.values() {
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
        if let Some(def_id) = opt_def_id {
            map.insert(def_id, symbol);
        }
    }

    map
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { trimmed_def_paths, ..*providers };
}
