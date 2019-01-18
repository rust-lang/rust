use crate::hir;
use crate::hir::def::Namespace;
use crate::hir::map::DefPathData;
use crate::hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use crate::middle::cstore::{ExternCrate, ExternCrateSource};
use crate::middle::region;
use crate::ty::{self, DefIdTree, ParamConst, Ty, TyCtxt, TypeFoldable};
use crate::ty::subst::{Kind, Subst, SubstsRef, UnpackedKind};
use crate::mir::interpret::ConstValue;
use syntax::symbol::{keywords, Symbol};

use syntax::symbol::InternedString;

use std::cell::Cell;
use std::fmt::{self, Write as _};
use std::iter;

// `pretty` is a separate module only for organization.
use super::*;

macro_rules! nest {
    ($closure:expr) => {
        scoped_cx!() = scoped_cx!().nest($closure)?
    }
}
macro_rules! print_inner {
    (write ($($data:expr),+)) => {
        write!(scoped_cx!().printer, $($data),+)?
    };
    ($kind:ident ($data:expr)) => {
        nest!(|cx| $data.$kind(cx))
    };
}
macro_rules! p {
    ($($kind:ident $data:tt),+) => {
        {
            $(print_inner!($kind $data));+
        }
    };
}
macro_rules! define_scoped_cx {
    ($cx:ident) => {
        #[allow(unused_macros)]
        macro_rules! scoped_cx {
            () => ($cx)
        }
    };
}

thread_local! {
    static FORCE_IMPL_FILENAME_LINE: Cell<bool> = Cell::new(false);
    static SHOULD_PREFIX_WITH_CRATE: Cell<bool> = Cell::new(false);
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
    pub fn highlighting_region(
        &mut self,
        region: ty::Region<'_>,
        number: usize,
    ) {
        let num_slots = self.highlight_regions.len();
        let first_avail_slot = self.highlight_regions.iter_mut()
            .filter(|s| s.is_none())
            .next()
            .unwrap_or_else(|| {
                bug!(
                    "can only highlight {} placeholders at a time",
                    num_slots,
                )
            });
        *first_avail_slot = Some((*region, number));
    }

    /// Convenience wrapper for `highlighting_region`.
    pub fn highlighting_region_vid(
        &mut self,
        vid: ty::RegionVid,
        number: usize,
    ) {
        self.highlighting_region(&ty::ReVar(vid), number)
    }

    /// Returns `Some(n)` with the number to use for the given region, if any.
    fn region_highlighted(&self, region: ty::Region<'_>) -> Option<usize> {
        self
            .highlight_regions
            .iter()
            .filter_map(|h| match h {
                Some((r, n)) if r == region => Some(*n),
                _ => None,
            })
            .next()
    }

    /// Highlight the given bound region.
    /// We can only highlight one bound region at a time. See
    /// the field `highlight_bound_region` for more detailed notes.
    pub fn highlighting_bound_region(
        &mut self,
        br: ty::BoundRegion,
        number: usize,
    ) {
        assert!(self.highlight_bound_region.is_none());
        self.highlight_bound_region = Some((br, number));
    }
}

/// Trait for printers that pretty-print using `fmt::Write` to the printer.
pub trait PrettyPrinter:
    Printer<
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
    > +
    fmt::Write
{
    /// Enter a nested print context, for pretty-printing
    /// nested components in some larger context.
    fn nest<'a, 'gcx, 'tcx, E>(
        self: PrintCx<'a, 'gcx, 'tcx, Self>,
        f: impl FnOnce(PrintCx<'_, 'gcx, 'tcx, Self>) -> Result<Self, E>,
    ) -> Result<PrintCx<'a, 'gcx, 'tcx, Self>, E> {
        let printer = f(PrintCx {
            tcx: self.tcx,
            printer: self.printer,
            config: self.config,
        })?;
        Ok(PrintCx {
            tcx: self.tcx,
            printer,
            config: self.config,
        })
    }

    /// Like `print_def_path` but for value paths.
    fn print_value_path(
        self: PrintCx<'_, '_, 'tcx, Self>,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.print_def_path(def_id, substs, iter::empty())
    }

    /// Print `<...>` around what `f` prints.
    fn generic_delimiters<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        f: impl FnOnce(PrintCx<'_, 'gcx, 'tcx, Self>) -> Result<Self, Self::Error>,
    ) -> Result<Self, Self::Error>;

    /// Return `true` if the region should be printed in path generic args
    /// even when it's `'_`, such as in e.g. `Foo<'_, '_, '_>`.
    fn always_print_region_in_paths(
        self: &PrintCx<'_, '_, '_, Self>,
        _region: ty::Region<'_>,
    ) -> bool {
        false
    }

    // HACK(eddyb) Trying to print a lifetime might not print anything, which
    // may need special handling in the caller (of `ty::RegionKind::print`).
    // To avoid printing to a temporary string (which isn't even supported),
    // the `print_region_outputs_anything` method can instead be used to
    // determine this, ahead of time.
    //
    // NB: this must be kept in sync with the implementation of `print_region`.
    fn print_region_outputs_anything(
        self: &PrintCx<'_, '_, '_, Self>,
        region: ty::Region<'_>,
    ) -> bool;
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    // HACK(eddyb) get rid of `def_path_str` and/or pass `Namespace` explicitly always
    // (but also some things just print a `DefId` generally so maybe we need this?)
    fn guess_def_namespace(self, def_id: DefId) -> Namespace {
        match self.def_key(def_id).disambiguated_data.data {
            DefPathData::ValueNs(..) |
            DefPathData::EnumVariant(..) |
            DefPathData::Field(..) |
            DefPathData::AnonConst |
            DefPathData::ConstParam(..) |
            DefPathData::ClosureExpr |
            DefPathData::StructCtor => Namespace::ValueNS,

            DefPathData::MacroDef(..) => Namespace::MacroNS,

            _ => Namespace::TypeNS,
        }
    }

    /// Returns a string identifying this `DefId. This string is
    /// suitable for user output.
    pub fn def_path_str(self, def_id: DefId) -> String {
        let ns = self.guess_def_namespace(def_id);
        debug!("def_path_str: def_id={:?}, ns={:?}", def_id, ns);
        let mut s = String::new();
        let _ = PrintCx::with(self, FmtPrinter::new(&mut s, ns), |cx| {
            cx.print_def_path(def_id, None, iter::empty())
        });
        s
    }
}

pub struct FmtPrinter<F: fmt::Write> {
    fmt: F,
    empty: bool,
    in_value: bool,
    pub region_highlight_mode: RegionHighlightMode,
}

impl<F: fmt::Write> FmtPrinter<F> {
    pub fn new(fmt: F, ns: Namespace) -> Self {
        FmtPrinter {
            fmt,
            empty: true,
            in_value: ns == Namespace::ValueNS,
            region_highlight_mode: RegionHighlightMode::default(),
        }
    }
}

impl<'gcx, 'tcx, P: PrettyPrinter> PrintCx<'_, 'gcx, 'tcx, P> {
    /// If possible, this returns a global path resolving to `def_id` that is visible
    /// from at least one local module and returns true. If the crate defining `def_id` is
    /// declared with an `extern crate`, the path is guaranteed to use the `extern crate`.
    fn try_print_visible_def_path(
        mut self,
        def_id: DefId,
    ) -> Result<(P, bool), P::Error> {
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
                    debug!("try_print_visible_def_path: def_id={:?}", def_id);
                    return Ok((if !span.is_dummy() {
                        self.print_def_path(def_id, None, iter::empty())?
                    } else {
                        self.path_crate(cnum)?
                    }, true));
                }
                None => {
                    return Ok((self.path_crate(cnum)?, true));
                }
                _ => {},
            }
        }

        if def_id.is_local() {
            return Ok((self.printer, false));
        }

        let visible_parent_map = self.tcx.visible_parent_map(LOCAL_CRATE);

        let mut cur_def_key = self.tcx.def_key(def_id);
        debug!("try_print_visible_def_path: cur_def_key={:?}", cur_def_key);

        // For a UnitStruct or TupleStruct we want the name of its parent rather than <unnamed>.
        if let DefPathData::StructCtor = cur_def_key.disambiguated_data.data {
            let parent = DefId {
                krate: def_id.krate,
                index: cur_def_key.parent.expect("DefPathData::StructCtor missing a parent"),
            };

            cur_def_key = self.tcx.def_key(parent);
        }

        let visible_parent = match visible_parent_map.get(&def_id).cloned() {
            Some(parent) => parent,
            None => return Ok((self.printer, false)),
        };
        // HACK(eddyb) this uses `nest` to avoid knowing ahead of time whether
        // the entire path will succeed or not. To support printers that do not
        // implement `PrettyPrinter`, a `Vec` or linked list on the stack would
        // need to be built, before starting to print anything.
        let mut prefix_success = false;
        nest!(|cx| {
            let (printer, success) = cx.try_print_visible_def_path(visible_parent)?;
            prefix_success = success;
            Ok(printer)
        });
        if !prefix_success {
            return Ok((self.printer, false));
        };
        let actual_parent = self.tcx.parent(def_id);
        debug!(
            "try_print_visible_def_path: visible_parent={:?} actual_parent={:?}",
            visible_parent, actual_parent,
        );

        let data = cur_def_key.disambiguated_data.data;
        debug!(
            "try_print_visible_def_path: data={:?} visible_parent={:?} actual_parent={:?}",
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
        debug!("try_print_visible_def_path: symbol={:?}", symbol);
        Ok((self.path_append(|cx| Ok(cx.printer), &symbol)?, true))
    }

    pub fn pretty_path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<P::Path, P::Error> {
        if trait_ref.is_none() {
            // Inherent impls. Try to print `Foo::bar` for an inherent
            // impl on `Foo`, but fallback to `<Foo>::bar` if self-type is
            // anything other than a simple path.
            match self_ty.sty {
                ty::Adt(..) | ty::Foreign(_) |
                ty::Bool | ty::Char | ty::Str |
                ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
                    return self_ty.print_display(self);
                }

                _ => {}
            }
        }

        self.generic_delimiters(|mut cx| {
            define_scoped_cx!(cx);

            p!(print_display(self_ty));
            if let Some(trait_ref) = trait_ref {
                p!(write(" as "), print_display(trait_ref));
            }
            Ok(cx.printer)
        })
    }

    pub fn pretty_path_append_impl(
        mut self,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, P>,
        ) -> Result<P::Path, P::Error>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<P::Path, P::Error> {
        self = self.nest(print_prefix)?;

        self.generic_delimiters(|mut cx| {
            define_scoped_cx!(cx);

            p!(write("impl "));
            if let Some(trait_ref) = trait_ref {
                p!(print_display(trait_ref), write(" for "));
            }
            p!(print_display(self_ty));

            Ok(cx.printer)
        })
    }

    pub fn pretty_path_generic_args(
        mut self,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, P>,
        ) -> Result<P::Path, P::Error>,
        params: &[ty::GenericParamDef],
        substs: SubstsRef<'tcx>,
        projections: impl Iterator<Item = ty::ExistentialProjection<'tcx>>,
    ) -> Result<P::Path, P::Error> {
        self = self.nest(print_prefix)?;

        // Don't print `'_` if there's no printed region.
        let print_regions = params.iter().any(|param| {
            match substs[param.index as usize].unpack() {
                UnpackedKind::Lifetime(r) => {
                    self.always_print_region_in_paths(r) ||
                    self.print_region_outputs_anything(r)
                }
                _ => false,
            }
        });

        // Don't print args that are the defaults of their respective parameters.
        let num_supplied_defaults = if self.tcx.sess.verbose() {
            0
        } else {
            params.iter().rev().take_while(|param| {
                match param.kind {
                    ty::GenericParamDefKind::Lifetime => false,
                    ty::GenericParamDefKind::Type { has_default, .. } => {
                        has_default && substs[param.index as usize] == Kind::from(
                            self.tcx.type_of(param.def_id).subst(self.tcx, substs)
                        )
                    }
                    ty::GenericParamDefKind::Const => false, // FIXME(const_generics:defaults)
                }
            }).count()
        };

        let params = &params[..params.len() - num_supplied_defaults];
        let mut args = params.iter().map(|param| {
            substs[param.index as usize]
        }).filter(|arg| {
            match arg.unpack() {
                UnpackedKind::Lifetime(_) => print_regions,
                _ => true,
            }
        });
        let arg0 = args.next();

        let mut projections = projections;
        let projection0 = projections.next();

        if arg0.is_none() && projection0.is_none() {
            return Ok(self.printer);
        }

        self.generic_delimiters(|mut cx| {
            define_scoped_cx!(cx);

            let mut empty = true;
            let mut maybe_comma = |cx: &mut Self| {
                if empty {
                    empty = false;
                    Ok(())
                } else {
                    write!(cx.printer, ", ")
                }
            };

            for arg in arg0.into_iter().chain(args) {
                maybe_comma(&mut cx)?;

                if let UnpackedKind::Lifetime(region) = arg.unpack() {
                    if !cx.print_region_outputs_anything(region) {
                        // This happens when the value of the region
                        // parameter is not easily serialized. This may be
                        // because the user omitted it in the first place,
                        // or because it refers to some block in the code,
                        // etc. I'm not sure how best to serialize this.
                        p!(write("'_"));

                        continue;
                    }
                }

                p!(print_display(arg));
            }

            for projection in projection0.into_iter().chain(projections) {
                maybe_comma(&mut cx)?;

                p!(write("{}=", cx.tcx.associated_item(projection.item_def_id).ident),
                   print_display(projection.ty));
            }

            Ok(cx.printer)
        })
    }
}

impl<F: fmt::Write> fmt::Write for FmtPrinter<F> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.empty &= s.is_empty();
        self.fmt.write_str(s)
    }
}

impl<F: fmt::Write> Printer for FmtPrinter<F> {
    type Error = fmt::Error;

    type Path = Self;
    type Region = Self;
    type Type = Self;

    fn print_def_path(
        mut self: PrintCx<'_, '_, 'tcx, Self>,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        projections: impl Iterator<Item = ty::ExistentialProjection<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        // FIXME(eddyb) avoid querying `tcx.generics_of` and `tcx.def_key`
        // both here and in `default_print_def_path`.
        let generics = substs.map(|_| self.tcx.generics_of(def_id));
        if generics.as_ref().and_then(|g| g.parent).is_none() {
            let mut visible_path_success = false;
            self = self.nest(|cx| {
                let (printer, success) = cx.try_print_visible_def_path(def_id)?;
                visible_path_success = success;
                Ok(printer)
            })?;
            if visible_path_success {
                return if let (Some(generics), Some(substs)) = (generics, substs) {
                    let has_own_self = generics.has_self && generics.parent_count == 0;
                    let params = &generics.params[has_own_self as usize..];
                    self.path_generic_args(|cx| Ok(cx.printer), params, substs, projections)
                } else {
                    Ok(self.printer)
                };
            }
        }

        let key = self.tcx.def_key(def_id);
        if let DefPathData::Impl = key.disambiguated_data.data {
            // Always use types for non-local impls, where types are always
            // available, and filename/line-number is mostly uninteresting.
            let use_types =
                !def_id.is_local() || {
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
                return self.path_append(
                    |cx| cx.print_def_path(parent_def_id, None, iter::empty()),
                    &format!("<impl at {:?}>", span),
                );
            }
        }

        self.default_print_def_path(def_id, substs, projections)
    }

    fn print_region(
        self: PrintCx<'_, '_, '_, Self>,
        region: ty::Region<'_>,
    ) -> Result<Self::Region, Self::Error> {
        self.pretty_print_region(region)
    }

    fn print_type(
        self: PrintCx<'_, '_, 'tcx, Self>,
        ty: Ty<'tcx>,
    ) -> Result<Self::Type, Self::Error> {
        self.pretty_print_type(ty)
    }

    fn path_crate(
        mut self: PrintCx<'_, '_, '_, Self>,
        cnum: CrateNum,
    ) -> Result<Self::Path, Self::Error> {
        if cnum == LOCAL_CRATE {
            if self.tcx.sess.rust_2018() {
                // We add the `crate::` keyword on Rust 2018, only when desired.
                if SHOULD_PREFIX_WITH_CRATE.with(|flag| flag.get()) {
                    write!(self.printer, "{}", keywords::Crate.name())?;
                }
            }
            Ok(self.printer)
        } else {
            write!(self.printer, "{}", self.tcx.crate_name(cnum))?;
            Ok(self.printer)
        }
    }
    fn path_qualified(
        self: PrintCx<'_, '_, 'tcx, Self>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.pretty_path_qualified(self_ty, trait_ref)
    }

    fn path_append_impl<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, Self>,
        ) -> Result<Self::Path, Self::Error>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.pretty_path_append_impl(|cx| {
            let mut printer = print_prefix(cx)?;

            // HACK(eddyb) this accounts for `generic_delimiters`
            // printing `::<` instead of `<` if `in_value` is set.
            if !printer.empty && !printer.in_value {
                write!(printer, "::")?;
            }

            Ok(printer)
        }, self_ty, trait_ref)
    }
    fn path_append<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, Self>,
        ) -> Result<Self::Path, Self::Error>,
        text: &str,
    ) -> Result<Self::Path, Self::Error> {
        let mut printer = print_prefix(self)?;

        // FIXME(eddyb) `text` should never be empty, but it
        // currently is for `extern { ... }` "foreign modules".
        if !text.is_empty() {
            if !printer.empty {
                write!(printer, "::")?;
            }
            write!(printer, "{}", text)?;
        }

        Ok(printer)
    }
    fn path_generic_args<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, Self>,
        ) -> Result<Self::Path, Self::Error>,
        params: &[ty::GenericParamDef],
        substs: SubstsRef<'tcx>,
        projections: impl Iterator<Item = ty::ExistentialProjection<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.pretty_path_generic_args(print_prefix, params, substs, projections)
    }
}

impl<F: fmt::Write> PrettyPrinter for FmtPrinter<F> {
    fn nest<'a, 'gcx, 'tcx, E>(
        mut self: PrintCx<'a, 'gcx, 'tcx, Self>,
        f: impl FnOnce(PrintCx<'_, 'gcx, 'tcx, Self>) -> Result<Self, E>,
    ) -> Result<PrintCx<'a, 'gcx, 'tcx, Self>, E> {
        let was_empty = std::mem::replace(&mut self.printer.empty, true);
        let mut printer = f(PrintCx {
            tcx: self.tcx,
            printer: self.printer,
            config: self.config,
        })?;
        printer.empty &= was_empty;
        Ok(PrintCx {
            tcx: self.tcx,
            printer,
            config: self.config,
        })
    }

    fn print_value_path(
        mut self: PrintCx<'_, '_, 'tcx, Self>,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        let was_in_value = std::mem::replace(&mut self.printer.in_value, true);
        let mut printer = self.print_def_path(def_id, substs, iter::empty())?;
        printer.in_value = was_in_value;

        Ok(printer)
    }

    fn generic_delimiters<'gcx, 'tcx>(
        mut self: PrintCx<'_, 'gcx, 'tcx, Self>,
        f: impl FnOnce(PrintCx<'_, 'gcx, 'tcx, Self>) -> Result<Self, Self::Error>,
    ) -> Result<Self, Self::Error> {
        if !self.printer.empty && self.printer.in_value {
            write!(self.printer, "::<")?;
        } else {
            write!(self.printer, "<")?;
        }

        let was_in_value = std::mem::replace(&mut self.printer.in_value, false);
        let mut printer = f(self)?;
        printer.in_value = was_in_value;

        write!(printer, ">")?;
        Ok(printer)
    }

    fn always_print_region_in_paths(
        self: &PrintCx<'_, '_, '_, Self>,
        region: ty::Region<'_>,
    ) -> bool {
        *region != ty::ReErased
    }

    fn print_region_outputs_anything(
        self: &PrintCx<'_, '_, '_, Self>,
        region: ty::Region<'_>,
    ) -> bool {
        let highlight = self.printer.region_highlight_mode;
        if highlight.region_highlighted(region).is_some() {
            return true;
        }

        if self.tcx.sess.verbose() {
            return true;
        }

        let identify_regions = self.tcx.sess.opts.debugging_opts.identify_regions;

        match *region {
            ty::ReEarlyBound(ref data) => {
                data.name != "" && data.name != "'_"
            }

            ty::ReLateBound(_, br) |
            ty::ReFree(ty::FreeRegion { bound_region: br, .. }) |
            ty::RePlaceholder(ty::Placeholder { name: br, .. }) => {
                if let ty::BrNamed(_, name) = br {
                    if name != "" && name != "'_" {
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

            ty::ReScope(_) |
            ty::ReVar(_) if identify_regions => true,

            ty::ReVar(_) |
            ty::ReScope(_) |
            ty::ReErased => false,

            ty::ReStatic |
            ty::ReEmpty |
            ty::ReClosureBound(_) => true,
        }
    }
}

// HACK(eddyb) limited to `FmtPrinter` because of `region_highlight_mode`.
impl<F: fmt::Write> FmtPrinter<F> {
    pub fn pretty_print_region(
        mut self: PrintCx<'_, '_, '_, Self>,
        region: ty::Region<'_>,
    ) -> Result<Self, fmt::Error> {
        define_scoped_cx!(self);

        // Watch out for region highlights.
        let highlight = self.printer.region_highlight_mode;
        if let Some(n) = highlight.region_highlighted(region) {
            p!(write("'{}", n));
            return Ok(self.printer);
        }

        if self.tcx.sess.verbose() {
            return region.print_debug(self);
        }

        let identify_regions = self.tcx.sess.opts.debugging_opts.identify_regions;

        // These printouts are concise.  They do not contain all the information
        // the user might want to diagnose an error, but there is basically no way
        // to fit that into a short string.  Hence the recommendation to use
        // `explain_region()` or `note_and_explain_region()`.
        match *region {
            ty::ReEarlyBound(ref data) => {
                if data.name != "'_" {
                    p!(write("{}", data.name));
                }
            }
            ty::ReLateBound(_, br) |
            ty::ReFree(ty::FreeRegion { bound_region: br, .. }) |
            ty::RePlaceholder(ty::Placeholder { name: br, .. }) => {
                if let ty::BrNamed(_, name) = br {
                    if name != "" && name != "'_" {
                        p!(write("{}", name));
                        return Ok(self.printer);
                    }
                }

                if let Some((region, counter)) = highlight.highlight_bound_region {
                    if br == region {
                        p!(write("'{}", counter));
                    }
                }
            }
            ty::ReScope(scope) if identify_regions => {
                match scope.data {
                    region::ScopeData::Node =>
                        p!(write("'{}s", scope.item_local_id().as_usize())),
                    region::ScopeData::CallSite =>
                        p!(write("'{}cs", scope.item_local_id().as_usize())),
                    region::ScopeData::Arguments =>
                        p!(write("'{}as", scope.item_local_id().as_usize())),
                    region::ScopeData::Destruction =>
                        p!(write("'{}ds", scope.item_local_id().as_usize())),
                    region::ScopeData::Remainder(first_statement_index) => p!(write(
                        "'{}_{}rs",
                        scope.item_local_id().as_usize(),
                        first_statement_index.index()
                    )),
                }
            }
            ty::ReVar(region_vid) if identify_regions => {
                p!(write("{:?}", region_vid));
            }
            ty::ReVar(_) => {}
            ty::ReScope(_) |
            ty::ReErased => {}
            ty::ReStatic => p!(write("'static")),
            ty::ReEmpty => p!(write("'<empty>")),

            // The user should never encounter these in unsubstituted form.
            ty::ReClosureBound(vid) => p!(write("{:?}", vid)),
        }

        Ok(self.printer)
    }
}

impl<'gcx, 'tcx, P: PrettyPrinter> PrintCx<'_, 'gcx, 'tcx, P> {
    pub fn pretty_print_type(
        mut self,
        ty: Ty<'tcx>,
    ) -> Result<P::Type, P::Error> {
        define_scoped_cx!(self);

        match ty.sty {
            ty::Bool => p!(write("bool")),
            ty::Char => p!(write("char")),
            ty::Int(t) => p!(write("{}", t.ty_to_string())),
            ty::Uint(t) => p!(write("{}", t.ty_to_string())),
            ty::Float(t) => p!(write("{}", t.ty_to_string())),
            ty::RawPtr(ref tm) => {
                p!(write("*{} ", match tm.mutbl {
                    hir::MutMutable => "mut",
                    hir::MutImmutable => "const",
                }));
                p!(print(tm.ty))
            }
            ty::Ref(r, ty, mutbl) => {
                p!(write("&"));
                if self.print_region_outputs_anything(r) {
                    p!(print_display(r), write(" "));
                }
                p!(print(ty::TypeAndMut { ty, mutbl }))
            }
            ty::Never => p!(write("!")),
            ty::Tuple(ref tys) => {
                p!(write("("));
                let mut tys = tys.iter();
                if let Some(&ty) = tys.next() {
                    p!(print(ty), write(","));
                    if let Some(&ty) = tys.next() {
                        p!(write(" "), print(ty));
                        for &ty in tys {
                            p!(write(", "), print(ty));
                        }
                    }
                }
                p!(write(")"))
            }
            ty::FnDef(def_id, substs) => {
                let sig = self.tcx.fn_sig(def_id).subst(self.tcx, substs);
                p!(print(sig), write(" {{"));
                nest!(|cx| cx.print_value_path(def_id, Some(substs)));
                p!(write("}}"))
            }
            ty::FnPtr(ref bare_fn) => {
                p!(print(bare_fn))
            }
            ty::Infer(infer_ty) => p!(write("{}", infer_ty)),
            ty::Error => p!(write("[type error]")),
            ty::Param(ref param_ty) => p!(write("{}", param_ty)),
            ty::Bound(debruijn, bound_ty) => {
                match bound_ty.kind {
                    ty::BoundTyKind::Anon => {
                        if debruijn == ty::INNERMOST {
                            p!(write("^{}", bound_ty.var.index()))
                        } else {
                            p!(write("^{}_{}", debruijn.index(), bound_ty.var.index()))
                        }
                    }

                    ty::BoundTyKind::Param(p) => p!(write("{}", p)),
                }
            }
            ty::Adt(def, substs) => {
                nest!(|cx| cx.print_def_path(def.did, Some(substs), iter::empty()));
            }
            ty::Dynamic(data, r) => {
                let print_r = self.print_region_outputs_anything(r);
                if print_r {
                    p!(write("("));
                }
                p!(write("dyn "), print(data));
                if print_r {
                    p!(write(" + "), print_display(r), write(")"));
                }
            }
            ty::Foreign(def_id) => {
                nest!(|cx| cx.print_def_path(def_id, None, iter::empty()));
            }
            ty::Projection(ref data) => p!(print(data)),
            ty::UnnormalizedProjection(ref data) => {
                p!(write("Unnormalized("), print(data), write(")"))
            }
            ty::Placeholder(placeholder) => {
                p!(write("Placeholder({:?})", placeholder))
            }
            ty::Opaque(def_id, substs) => {
                if self.tcx.sess.verbose() {
                    p!(write("Opaque({:?}, {:?})", def_id, substs));
                    return Ok(self.printer);
                }

                let def_key = self.tcx.def_key(def_id);
                if let Some(name) = def_key.disambiguated_data.data.get_opt_name() {
                    p!(write("{}", name));
                    let mut substs = substs.iter();
                    // FIXME(eddyb) print this with `print_def_path`.
                    if let Some(first) = substs.next() {
                        p!(write("::<"));
                        p!(print_display(first));
                        for subst in substs {
                            p!(write(", "), print_display(subst));
                        }
                        p!(write(">"));
                    }
                    return Ok(self.printer);
                }
                // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                // by looking up the projections associated with the def_id.
                let bounds = self.tcx.predicates_of(def_id).instantiate(self.tcx, substs);

                let mut first = true;
                let mut is_sized = false;
                p!(write("impl"));
                for predicate in bounds.predicates {
                    if let Some(trait_ref) = predicate.to_opt_poly_trait_ref() {
                        // Don't print +Sized, but rather +?Sized if absent.
                        if Some(trait_ref.def_id()) == self.tcx.lang_items().sized_trait() {
                            is_sized = true;
                            continue;
                        }

                        p!(
                                write("{}", if first { " " } else { "+" }),
                                print(trait_ref));
                        first = false;
                    }
                }
                if !is_sized {
                    p!(write("{}?Sized", if first { " " } else { "+" }));
                } else if first {
                    p!(write(" Sized"));
                }
            }
            ty::Str => p!(write("str")),
            ty::Generator(did, substs, movability) => {
                let upvar_tys = substs.upvar_tys(did, self.tcx);
                let witness = substs.witness(did, self.tcx);
                if movability == hir::GeneratorMovability::Movable {
                    p!(write("[generator"));
                } else {
                    p!(write("[static generator"));
                }

                // FIXME(eddyb) should use `def_span`.
                if let Some(hir_id) = self.tcx.hir().as_local_hir_id(did) {
                    p!(write("@{:?}", self.tcx.hir().span_by_hir_id(hir_id)));
                    let mut sep = " ";
                    for (freevar, upvar_ty) in self.tcx.freevars(did)
                        .as_ref()
                        .map_or(&[][..], |fv| &fv[..])
                        .iter()
                        .zip(upvar_tys)
                    {
                        p!(
                            write("{}{}:",
                                    sep,
                                    self.tcx.hir().name(freevar.var_id())),
                            print(upvar_ty));
                        sep = ", ";
                    }
                } else {
                    // cross-crate closure types should only be
                    // visible in codegen bug reports, I imagine.
                    p!(write("@{:?}", did));
                    let mut sep = " ";
                    for (index, upvar_ty) in upvar_tys.enumerate() {
                        p!(
                                write("{}{}:", sep, index),
                                print(upvar_ty));
                        sep = ", ";
                    }
                }

                p!(write(" "), print(witness), write("]"))
            },
            ty::GeneratorWitness(types) => {
                nest!(|cx| cx.pretty_in_binder(&types))
            }
            ty::Closure(did, substs) => {
                let upvar_tys = substs.upvar_tys(did, self.tcx);
                p!(write("[closure"));

                // FIXME(eddyb) should use `def_span`.
                if let Some(hir_id) = self.tcx.hir().as_local_hir_id(did) {
                    if self.tcx.sess.opts.debugging_opts.span_free_formats {
                        p!(write("@{:?}", hir_id));
                    } else {
                        p!(write("@{:?}", self.tcx.hir().span_by_hir_id(hir_id)));
                    }
                    let mut sep = " ";
                    for (freevar, upvar_ty) in self.tcx.freevars(did)
                        .as_ref()
                        .map_or(&[][..], |fv| &fv[..])
                        .iter()
                        .zip(upvar_tys)
                    {
                        p!(
                            write("{}{}:",
                                    sep,
                                    self.tcx.hir().name(freevar.var_id())),
                            print(upvar_ty));
                        sep = ", ";
                    }
                } else {
                    // cross-crate closure types should only be
                    // visible in codegen bug reports, I imagine.
                    p!(write("@{:?}", did));
                    let mut sep = " ";
                    for (index, upvar_ty) in upvar_tys.enumerate() {
                        p!(
                                write("{}{}:", sep, index),
                                print(upvar_ty));
                        sep = ", ";
                    }
                }

                if self.config.is_verbose {
                    p!(write(
                        " closure_kind_ty={:?} closure_sig_ty={:?}",
                        substs.closure_kind_ty(did, self.tcx),
                        substs.closure_sig_ty(did, self.tcx)
                    ));
                }

                p!(write("]"))
            },
            ty::Array(ty, sz) => {
                p!(write("["), print(ty), write("; "));
                match sz {
                    ty::LazyConst::Unevaluated(_def_id, _substs) => {
                        p!(write("_"));
                    }
                    ty::LazyConst::Evaluated(c) => {
                        match c.val {
                            ConstValue::Infer(..) => p!(write("_")),
                            ConstValue::Param(ParamConst { name, .. }) =>
                                p!(write("{}", name)),
                            _ => p!(write("{}", c.unwrap_usize(self.tcx))),
                        }
                    }
                }
                p!(write("]"))
            }
            ty::Slice(ty) => {
                p!(write("["), print(ty), write("]"))
            }
        }

        Ok(self.printer)
    }

    pub fn pretty_fn_sig(
        mut self,
        inputs: &[Ty<'tcx>],
        c_variadic: bool,
        output: Ty<'tcx>,
    ) -> Result<P, fmt::Error> {
        define_scoped_cx!(self);

        p!(write("("));
        let mut inputs = inputs.iter();
        if let Some(&ty) = inputs.next() {
            p!(print_display(ty));
            for &ty in inputs {
                p!(write(", "), print_display(ty));
            }
            if c_variadic {
                p!(write(", ..."));
            }
        }
        p!(write(")"));
        if !output.is_unit() {
            p!(write(" -> "), print_display(output));
        }

        Ok(self.printer)
    }

    pub fn pretty_in_binder<T>(mut self, value: &ty::Binder<T>) -> Result<P, fmt::Error>
        where T: Print<'tcx, P, Output = P, Error = fmt::Error> + TypeFoldable<'tcx>
    {
        fn name_by_region_index(index: usize) -> InternedString {
            match index {
                0 => Symbol::intern("'r"),
                1 => Symbol::intern("'s"),
                i => Symbol::intern(&format!("'t{}", i-2)),
            }.as_interned_str()
        }

        // Replace any anonymous late-bound regions with named
        // variants, using gensym'd identifiers, so that we can
        // clearly differentiate between named and unnamed regions in
        // the output. We'll probably want to tweak this over time to
        // decide just how much information to give.
        if self.config.binder_depth == 0 {
            self.prepare_late_bound_region_info(value);
        }

        let mut empty = true;
        let mut start_or_continue = |cx: &mut Self, start: &str, cont: &str| {
            write!(cx.printer, "{}", if empty {
                empty = false;
                start
            } else {
                cont
            })
        };

        // NOTE(eddyb) this must be below `start_or_continue`'s definition
        // as that also has a `define_scoped_cx` and that kind of shadowing
        // is disallowed (name resolution thinks `scoped_cx!` is ambiguous).
        define_scoped_cx!(self);

        let old_region_index = self.config.region_index;
        let mut region_index = old_region_index;
        let new_value = self.tcx.replace_late_bound_regions(value, |br| {
            let _ = start_or_continue(&mut self, "for<", ", ");
            let br = match br {
                ty::BrNamed(_, name) => {
                    let _ = write!(self.printer, "{}", name);
                    br
                }
                ty::BrAnon(_) |
                ty::BrFresh(_) |
                ty::BrEnv => {
                    let name = loop {
                        let name = name_by_region_index(region_index);
                        region_index += 1;
                        if !self.is_name_used(&name) {
                            break name;
                        }
                    };
                    let _ = write!(self.printer, "{}", name);
                    ty::BrNamed(DefId::local(CRATE_DEF_INDEX), name)
                }
            };
            self.tcx.mk_region(ty::ReLateBound(ty::INNERMOST, br))
        }).0;
        start_or_continue(&mut self, "", "> ")?;

        // Push current state to gcx, and restore after writing new_value.
        self.config.binder_depth += 1;
        self.config.region_index = region_index;
        let result = new_value.print_display(PrintCx {
            tcx: self.tcx,
            printer: self.printer,
            config: self.config,
        });
        self.config.region_index = old_region_index;
        self.config.binder_depth -= 1;
        result
    }

    fn is_name_used(&self, name: &InternedString) -> bool {
        match self.config.used_region_names {
            Some(ref names) => names.contains(name),
            None => false,
        }
    }
}
