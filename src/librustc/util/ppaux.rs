use crate::hir::def::Namespace;
use crate::hir::def_id::DefId;
use crate::middle::region;
use crate::ty::subst::{Kind, Subst, SubstsRef, UnpackedKind};
use crate::ty::{BrAnon, BrEnv, BrFresh, BrNamed};
use crate::ty::{Bool, Char, Adt};
use crate::ty::{Error, Str, Array, Slice, Float, FnDef, FnPtr};
use crate::ty::{Param, Bound, RawPtr, Ref, Never, Tuple};
use crate::ty::{Closure, Generator, GeneratorWitness, Foreign, Projection, Opaque};
use crate::ty::{Placeholder, UnnormalizedProjection, Dynamic, Int, Uint, Infer};
use crate::ty::{self, ParamConst, Ty, TypeFoldable};
use crate::ty::print::{FmtPrinter, PrettyPrinter, PrintCx, Print, Printer};
use crate::mir::interpret::ConstValue;

use std::cell::Cell;
use std::fmt::{self, Write as _};
use std::iter;
use std::usize;

use rustc_target::spec::abi::Abi;
use syntax::ast::CRATE_NODE_ID;
use syntax::symbol::{Symbol, InternedString};
use crate::hir;

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

thread_local! {
    /// Mechanism for highlighting of specific regions for display in NLL region inference errors.
    /// Contains region to highlight and counter for number to use when highlighting.
    static REGION_HIGHLIGHT_MODE: Cell<RegionHighlightMode> =
        Cell::new(RegionHighlightMode::default())
}

impl RegionHighlightMode {
    /// Reads and returns the current region highlight settings (accesses thread-local state).
    pub fn get() -> Self {
        REGION_HIGHLIGHT_MODE.with(|c| c.get())
    }

    // Internal helper to update current settings during the execution of `op`.
    fn set<R>(
        old_mode: Self,
        new_mode: Self,
        op: impl FnOnce() -> R,
    ) -> R {
        REGION_HIGHLIGHT_MODE.with(|c| {
            c.set(new_mode);
            let result = op();
            c.set(old_mode);
            result
        })
    }

    /// If `region` and `number` are both `Some`, invokes
    /// `highlighting_region`; otherwise, just invokes `op` directly.
    pub fn maybe_highlighting_region<R>(
        region: Option<ty::Region<'_>>,
        number: Option<usize>,
        op: impl FnOnce() -> R,
    ) -> R {
        if let Some(k) = region {
            if let Some(n) = number {
                return Self::highlighting_region(k, n, op);
            }
        }

        op()
    }

    /// During the execution of `op`, highlights the region inference
    /// variable `vid` as `'N`. We can only highlight one region `vid`
    /// at a time.
    pub fn highlighting_region<R>(
        region: ty::Region<'_>,
        number: usize,
        op: impl FnOnce() -> R,
    ) -> R {
        let old_mode = Self::get();
        let mut new_mode = old_mode;
        let first_avail_slot = new_mode.highlight_regions.iter_mut()
            .filter(|s| s.is_none())
            .next()
            .unwrap_or_else(|| {
                panic!(
                    "can only highlight {} placeholders at a time",
                    old_mode.highlight_regions.len(),
                )
            });
        *first_avail_slot = Some((*region, number));
        Self::set(old_mode, new_mode, op)
    }

    /// Convenience wrapper for `highlighting_region`.
    pub fn highlighting_region_vid<R>(
        vid: ty::RegionVid,
        number: usize,
        op: impl FnOnce() -> R,
    ) -> R {
        Self::highlighting_region(&ty::ReVar(vid), number, op)
    }

    /// Returns `Some(n)` with the number to use for the given region, if any.
    fn region_highlighted(&self, region: ty::Region<'_>) -> Option<usize> {
        Self::get()
            .highlight_regions
            .iter()
            .filter_map(|h| match h {
                Some((r, n)) if r == region => Some(*n),
                _ => None,
            })
            .next()
    }

    /// During the execution of `op`, highlight the given bound
    /// region. We can only highlight one bound region at a time. See
    /// the field `highlight_bound_region` for more detailed notes.
    pub fn highlighting_bound_region<R>(
        br: ty::BoundRegion,
        number: usize,
        op: impl FnOnce() -> R,
    ) -> R {
        let old_mode = Self::get();
        assert!(old_mode.highlight_bound_region.is_none());
        Self::set(
            old_mode,
            Self {
                highlight_bound_region: Some((br, number)),
                ..old_mode
            },
            op,
        )
    }

    /// Returns `Some(N)` if the placeholder `p` is highlighted to print as "`'N`".
    pub fn placeholder_highlight(&self, p: ty::PlaceholderRegion) -> Option<usize> {
        self.region_highlighted(&ty::RePlaceholder(p))
    }
}

macro_rules! gen_display_debug_body {
    ( $with:path ) => {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            PrintCx::with(FmtPrinter { fmt: f }, |mut cx| {
                $with(&cx.tcx.lift(self).expect("could not lift for printing"), &mut cx)
            })
        }
    };
}
macro_rules! gen_display_debug {
    ( ($($x:tt)+) $target:ty, display yes ) => {
        impl<$($x)+> fmt::Display for $target {
            gen_display_debug_body! { Print::print_display }
        }
    };
    ( () $target:ty, display yes ) => {
        impl fmt::Display for $target {
            gen_display_debug_body! { Print::print_display }
        }
    };
    ( ($($x:tt)+) $target:ty, debug yes ) => {
        impl<$($x)+> fmt::Debug for $target {
            gen_display_debug_body! { Print::print_debug }
        }
    };
    ( () $target:ty, debug yes ) => {
        impl fmt::Debug for $target {
            gen_display_debug_body! { Print::print_debug }
        }
    };
    ( $generic:tt $target:ty, $t:ident no ) => {};
}
macro_rules! gen_print_impl {
    ( ($($x:tt)+) $target:ty, ($self:ident, $cx:ident) $disp:block $dbg:block ) => {
        impl<$($x)+, P: PrettyPrinter> Print<'tcx, P> for $target {
            type Output = ();
            type Error = fmt::Error;
            fn print(&$self, $cx: &mut PrintCx<'_, '_, 'tcx, P>) -> fmt::Result {
                define_scoped_cx!($cx);
                if $cx.is_debug $dbg
                else $disp
            }
        }
    };
    ( () $target:ty, ($self:ident, $cx:ident) $disp:block $dbg:block ) => {
        impl<P: PrettyPrinter> Print<'tcx, P> for $target {
            type Output = ();
            type Error = fmt::Error;
            fn print(&$self, $cx: &mut PrintCx<'_, '_, 'tcx, P>) -> fmt::Result {
                define_scoped_cx!($cx);
                if $cx.is_debug $dbg
                else $disp
            }
        }
    };
    ( $generic:tt $target:ty,
      $vars:tt $gendisp:ident $disp:block $gendbg:ident $dbg:block ) => {
        gen_print_impl! { $generic $target, $vars $disp $dbg }
        gen_display_debug! { $generic $target, display $gendisp }
        gen_display_debug! { $generic $target, debug $gendbg }
    }
}
macro_rules! define_print {
    ( $generic:tt $target:ty,
      $vars:tt { display $disp:block debug $dbg:block } ) => {
        gen_print_impl! { $generic $target, $vars yes $disp yes $dbg }
    };
    ( $generic:tt $target:ty,
      $vars:tt { debug $dbg:block display $disp:block } ) => {
        gen_print_impl! { $generic $target, $vars yes $disp yes $dbg }
    };
    ( $generic:tt $target:ty,
      $vars:tt { debug $dbg:block } ) => {
        gen_print_impl! { $generic $target, $vars no {
            bug!(concat!("display not implemented for ", stringify!($target)));
        } yes $dbg }
    };
    ( $generic:tt $target:ty,
      ($self:ident, $cx:ident) { display $disp:block } ) => {
        gen_print_impl! { $generic $target, ($self, $cx) yes $disp no {
            write!($cx.printer, "{:?}", $self)
        } }
    };
}
macro_rules! define_print_multi {
    ( [ $($generic:tt $target:ty),* ] $vars:tt $def:tt ) => {
        $(define_print! { $generic $target, $vars $def })*
    };
}
macro_rules! print_inner {
    (write ($($data:expr),+)) => {
        write!(scoped_cx!().printer, $($data),+)
    };
    ($kind:ident ($data:expr)) => {
        $data.$kind(scoped_cx!())
    };
}
macro_rules! p {
    ($($kind:ident $data:tt),+) => {
        (|| -> fmt::Result {
            $(print_inner!($kind $data)?;)+
            Ok(())
        })()
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

impl<P: PrettyPrinter> PrintCx<'a, 'gcx, 'tcx, P> {
    fn fn_sig(
        &mut self,
        inputs: &[Ty<'tcx>],
        c_variadic: bool,
        output: Ty<'tcx>,
    ) -> fmt::Result {
        define_scoped_cx!(self);

        p!(write("("))?;
        let mut inputs = inputs.iter();
        if let Some(&ty) = inputs.next() {
            p!(print_display(ty))?;
            for &ty in inputs {
                p!(write(", "), print_display(ty))?;
            }
            if c_variadic {
                p!(write(", ..."))?;
            }
        }
        p!(write(")"))?;
        if !output.is_unit() {
            p!(write(" -> "), print_display(output))?;
        }

        Ok(())
    }

    fn in_binder<T>(&mut self, value: &ty::Binder<T>) -> Result<T::Output, fmt::Error>
        where T: Print<'tcx, P, Error = fmt::Error> + TypeFoldable<'tcx>
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
        if self.binder_depth == 0 {
            self.prepare_late_bound_region_info(value);
        }

        let mut empty = true;
        let mut start_or_continue = |cx: &mut Self, start: &str, cont: &str| {
            define_scoped_cx!(cx);

            if empty {
                empty = false;
                p!(write("{}", start))
            } else {
                p!(write("{}", cont))
            }
        };

        // NOTE(eddyb) this must be below `start_or_continue`'s definition
        // as that also has a `define_scoped_cx` and that kind of shadowing
        // is disallowed (name resolution thinks `scoped_cx!` is ambiguous).
        define_scoped_cx!(self);

        let old_region_index = self.region_index;
        let mut region_index = old_region_index;
        let new_value = self.tcx.replace_late_bound_regions(value, |br| {
            let _ = start_or_continue(self, "for<", ", ");
            let br = match br {
                ty::BrNamed(_, name) => {
                    let _ = p!(write("{}", name));
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
                    let _ = p!(write("{}", name));
                    ty::BrNamed(self.tcx.hir().local_def_id(CRATE_NODE_ID), name)
                }
            };
            self.tcx.mk_region(ty::ReLateBound(ty::INNERMOST, br))
        }).0;
        start_or_continue(self, "", "> ")?;

        // Push current state to gcx, and restore after writing new_value.
        self.binder_depth += 1;
        self.region_index = region_index;
        let result = new_value.print_display(self);
        self.region_index = old_region_index;
        self.binder_depth -= 1;
        result
    }

    fn is_name_used(&self, name: &InternedString) -> bool {
        match self.used_region_names {
            Some(ref names) => names.contains(name),
            None => false,
        }
    }
}

pub fn parameterized<F: fmt::Write>(
    f: &mut F,
    did: DefId,
    substs: SubstsRef<'_>,
    ns: Namespace,
) -> fmt::Result {
    PrintCx::with(FmtPrinter { fmt: f }, |mut cx| {
        let substs = cx.tcx.lift(&substs).expect("could not lift for printing");
        let _ = cx.print_def_path(did, Some(substs), ns, iter::empty())?;
        Ok(())
    })
}

define_print! {
    ('tcx) &'tcx ty::List<ty::ExistentialPredicate<'tcx>>, (self, cx) {
        display {
            // Generate the main trait ref, including associated types.
            let mut first = true;

            if let Some(principal) = self.principal() {
                let mut resugared_principal = false;

                // Special-case `Fn(...) -> ...` and resugar it.
                if !cx.is_verbose && cx.tcx.lang_items().fn_trait_kind(principal.def_id).is_some() {
                    if let Tuple(ref args) = principal.substs.type_at(0).sty {
                        let mut projections = self.projection_bounds();
                        if let (Some(proj), None) = (projections.next(), projections.next()) {
                            let _ = cx.print_def_path(
                                principal.def_id,
                                None,
                                Namespace::TypeNS,
                                iter::empty(),
                            )?;
                            cx.fn_sig(args, false, proj.ty)?;
                            resugared_principal = true;
                        }
                    }
                }

                if !resugared_principal {
                    // Use a type that can't appear in defaults of type parameters.
                    let dummy_self = cx.tcx.mk_infer(ty::FreshTy(0));
                    let principal = principal.with_self_ty(cx.tcx, dummy_self);
                    let _ = cx.print_def_path(
                        principal.def_id,
                        Some(principal.substs),
                        Namespace::TypeNS,
                        self.projection_bounds(),
                    )?;
                }
                first = false;
            }

            // Builtin bounds.
            // FIXME(eddyb) avoid printing twice (needed to ensure
            // that the auto traits are sorted *and* printed via cx).
            let mut auto_traits: Vec<_> = self.auto_traits().map(|did| {
                (cx.tcx.def_path_str(did), did)
            }).collect();

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
                    p!(write(" + "))?;
                }
                first = false;

                let _ = cx.print_def_path(
                    def_id,
                    None,
                    Namespace::TypeNS,
                    iter::empty(),
                )?;
            }

            Ok(())
        }
    }
}

impl fmt::Debug for ty::GenericParamDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_name = match self.kind {
            ty::GenericParamDefKind::Lifetime => "Lifetime",
            ty::GenericParamDefKind::Type { .. } => "Type",
            ty::GenericParamDefKind::Const => "Const",
        };
        write!(f, "{}({}, {:?}, {})",
               type_name,
               self.name,
               self.def_id,
               self.index)
    }
}

impl fmt::Debug for ty::TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrintCx::with(FmtPrinter { fmt: f }, |mut cx| {
            let _ = cx.print_def_path(
                self.def_id,
                None,
                Namespace::TypeNS,
                iter::empty(),
            )?;
            Ok(())
        })
    }
}

impl fmt::Debug for ty::AdtDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrintCx::with(FmtPrinter { fmt: f }, |mut cx| {
            let _ = cx.print_def_path(
                self.did,
                None,
                Namespace::TypeNS,
                iter::empty(),
            )?;
            Ok(())
        })
    }
}

impl<'tcx> fmt::Debug for ty::ClosureUpvar<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClosureUpvar({:?},{:?})",
               self.def,
               self.ty)
    }
}

impl fmt::Debug for ty::UpvarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrintCx::with(FmtPrinter { fmt: f }, |mut cx| {
            define_scoped_cx!(cx);
            p!(write("UpvarId({:?};`{}`;{:?})",
                self.var_path.hir_id,
                cx.tcx.hir().name_by_hir_id(self.var_path.hir_id),
                self.closure_expr_id))
        })
    }
}

impl<'tcx> fmt::Debug for ty::UpvarBorrow<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UpvarBorrow({:?}, {:?})",
               self.kind, self.region)
    }
}

define_print! {
    ('tcx) &'tcx ty::List<Ty<'tcx>>, (self, cx) {
        display {
            p!(write("{{"))?;
            let mut tys = self.iter();
            if let Some(&ty) = tys.next() {
                p!(print(ty))?;
                for &ty in tys {
                    p!(write(", "), print(ty))?;
                }
            }
            p!(write("}}"))
        }
    }
}

define_print! {
    ('tcx) ty::TypeAndMut<'tcx>, (self, cx) {
        display {
            p!(
                   write("{}", if self.mutbl == hir::MutMutable { "mut " } else { "" }),
                   print(self.ty))
        }
    }
}

define_print! {
    ('tcx) ty::ExistentialTraitRef<'tcx>, (self, cx) {
        display {
            let dummy_self = cx.tcx.mk_infer(ty::FreshTy(0));

            ty::Binder::bind(*self)
                .with_self_ty(cx.tcx, dummy_self)
                .skip_binder()
                .print_display(cx)
        }
        debug {
            self.print_display(cx)
        }
    }
}

define_print! {
    ('tcx) ty::adjustment::Adjustment<'tcx>, (self, cx) {
        debug {
            p!(write("{:?} -> ", self.kind), print(self.target))
        }
    }
}

define_print! {
    () ty::BoundRegion, (self, cx) {
        display {
            if cx.is_verbose {
                return self.print_debug(cx);
            }

            if let BrNamed(_, name) = *self {
                if name != "" && name != "'_" {
                    return p!(write("{}", name));
                }
            }

            let highlight = RegionHighlightMode::get();
            if let Some((region, counter)) = highlight.highlight_bound_region {
                if *self == region {
                    return p!(write("'{}", counter));
                }
            }

            Ok(())
        }
        debug {
            return match *self {
                BrAnon(n) => p!(write("BrAnon({:?})", n)),
                BrFresh(n) => p!(write("BrFresh({:?})", n)),
                BrNamed(did, name) => {
                    p!(write("BrNamed({:?}:{:?}, {})",
                           did.krate, did.index, name))
                }
                BrEnv => p!(write("BrEnv")),
            };
        }
    }
}

// HACK(eddyb) (see `ty::RegionKind::display_outputs_anything`)
//
// NB: this must be kept in sync with the printing logic above.
impl ty::BoundRegion {
    fn display_outputs_anything<P>(&self, cx: &mut PrintCx<'_, '_, '_, P>) -> bool {
        if cx.is_verbose {
            return true;
        }

        if let BrNamed(_, name) = *self {
            if name != "" && name != "'_" {
                return true;
            }
        }

        let highlight = RegionHighlightMode::get();
        if let Some((region, _)) = highlight.highlight_bound_region {
            if *self == region {
                return true;
            }
        }

        false
    }
}

define_print! {
    () ty::PlaceholderRegion, (self, cx) {
        display {
            if cx.is_verbose {
                return self.print_debug(cx);
            }

            let highlight = RegionHighlightMode::get();
            if let Some(counter) = highlight.placeholder_highlight(*self) {
                return p!(write("'{}", counter));
            }

            p!(print_display(self.name))
        }
    }
}

// HACK(eddyb) (see `ty::RegionKind::display_outputs_anything`)
//
// NB: this must be kept in sync with the printing logic above.
impl ty::PlaceholderRegion {
    fn display_outputs_anything<P>(&self, cx: &mut PrintCx<'_, '_, '_, P>) -> bool {
        if cx.is_verbose {
            return true;
        }

        let highlight = RegionHighlightMode::get();
        if highlight.placeholder_highlight(*self).is_some() {
            return true;
        }

        self.name.display_outputs_anything(cx)
    }
}

define_print! {
    () ty::RegionKind, (self, cx) {
        display {
            if cx.is_verbose {
                return self.print_debug(cx);
            }

            // Watch out for region highlights.
            if let Some(n) = RegionHighlightMode::get().region_highlighted(self) {
                return p!(write("'{:?}", n));
            }

            // These printouts are concise.  They do not contain all the information
            // the user might want to diagnose an error, but there is basically no way
            // to fit that into a short string.  Hence the recommendation to use
            // `explain_region()` or `note_and_explain_region()`.
            match *self {
                ty::ReEarlyBound(ref data) => {
                    if data.name != "'_" {
                        p!(write("{}", data.name))
                    } else {
                        Ok(())
                    }
                }
                ty::ReLateBound(_, br) |
                ty::ReFree(ty::FreeRegion { bound_region: br, .. }) => {
                    p!(print_display(br))
                }
                ty::RePlaceholder(p) => {
                    p!(print_display(p))
                }
                ty::ReScope(scope) if cx.identify_regions => {
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
                ty::ReVar(region_vid) if cx.identify_regions => {
                    p!(print_debug(region_vid))
                }
                ty::ReVar(region_vid) => {
                    p!(print_display(region_vid))
                }
                ty::ReScope(_) |
                ty::ReErased => Ok(()),
                ty::ReStatic => p!(write("'static")),
                ty::ReEmpty => p!(write("'<empty>")),

                // The user should never encounter these in unsubstituted form.
                ty::ReClosureBound(vid) => p!(write("{:?}", vid)),
            }
        }
        debug {
            match *self {
                ty::ReEarlyBound(ref data) => {
                    p!(write("ReEarlyBound({}, {})",
                           data.index,
                           data.name))
                }

                ty::ReClosureBound(ref vid) => {
                    p!(write("ReClosureBound({:?})",
                           vid))
                }

                ty::ReLateBound(binder_id, ref bound_region) => {
                    p!(write("ReLateBound({:?}, {:?})",
                           binder_id,
                           bound_region))
                }

                ty::ReFree(ref fr) => p!(write("{:?}", fr)),

                ty::ReScope(id) => {
                    p!(write("ReScope({:?})", id))
                }

                ty::ReStatic => p!(write("ReStatic")),

                ty::ReVar(ref vid) => {
                    p!(write("{:?}", vid))
                }

                ty::RePlaceholder(placeholder) => {
                    p!(write("RePlaceholder({:?})", placeholder))
                }

                ty::ReEmpty => p!(write("ReEmpty")),

                ty::ReErased => p!(write("ReErased"))
            }
        }
    }
}

// HACK(eddyb) Trying to print a lifetime might not print anything, which
// may need special handling in the caller (of `ty::RegionKind::print`).
// To avoid printing to a temporary string, the `display_outputs_anything`
// method can instead be used to determine this, ahead of time.
//
// NB: this must be kept in sync with the printing logic above.
impl ty::RegionKind {
    // HACK(eddyb) `pub(crate)` only for `ty::print`.
    pub(crate) fn display_outputs_anything<P>(&self, cx: &mut PrintCx<'_, '_, '_, P>) -> bool {
        if cx.is_verbose {
            return true;
        }

        if RegionHighlightMode::get().region_highlighted(self).is_some() {
            return true;
        }

        match *self {
            ty::ReEarlyBound(ref data) => {
                data.name != "" && data.name != "'_"
            }

            ty::ReLateBound(_, br) |
            ty::ReFree(ty::FreeRegion { bound_region: br, .. }) => {
                br.display_outputs_anything(cx)
            }

            ty::RePlaceholder(p) => p.display_outputs_anything(cx),

            ty::ReScope(_) |
            ty::ReVar(_) if cx.identify_regions => true,

            ty::ReVar(region_vid) => region_vid.display_outputs_anything(cx),

            ty::ReScope(_) |
            ty::ReErased => false,

            ty::ReStatic |
            ty::ReEmpty |
            ty::ReClosureBound(_) => true,
        }
    }
}

define_print! {
    () ty::FreeRegion, (self, cx) {
        debug {
            p!(write("ReFree({:?}, {:?})", self.scope, self.bound_region))
        }
    }
}

define_print! {
    () ty::Variance, (self, cx) {
        debug {
            cx.printer.write_str(match *self {
                ty::Covariant => "+",
                ty::Contravariant => "-",
                ty::Invariant => "o",
                ty::Bivariant => "*",
            })
        }
    }
}

define_print! {
    ('tcx) ty::FnSig<'tcx>, (self, cx) {
        display {
            if self.unsafety == hir::Unsafety::Unsafe {
                p!(write("unsafe "))?;
            }

            if self.abi != Abi::Rust {
                p!(write("extern {} ", self.abi))?;
            }

            p!(write("fn"))?;
            cx.fn_sig(self.inputs(), self.c_variadic, self.output())
        }
        debug {
            p!(write("({:?}; c_variadic: {})->{:?}",
                self.inputs(), self.c_variadic, self.output()))
        }
    }
}

impl fmt::Debug for ty::TyVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}t", self.index)
    }
}

impl<'tcx> fmt::Debug for ty::ConstVid<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}f", self.index)
    }
}

impl fmt::Debug for ty::IntVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}i", self.index)
    }
}

impl fmt::Debug for ty::FloatVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}f", self.index)
    }
}

define_print! {
    () ty::RegionVid, (self, cx) {
        display {
            if cx.is_verbose {
                return self.print_debug(cx);
            }

            let highlight = RegionHighlightMode::get();
            if let Some(counter) = highlight.region_highlighted(&ty::ReVar(*self)) {
                return p!(write("'{:?}", counter));
            }

            Ok(())
        }
        debug {
            // HACK(eddyb) this is duplicated from `display` printing,
            // to keep NLL borrowck working even with `-Zverbose`.
            let highlight = RegionHighlightMode::get();
            if let Some(counter) = highlight.region_highlighted(&ty::ReVar(*self)) {
                return p!(write("'{:?}", counter));
            }

            p!(write("'_#{}r", self.index()))
        }
    }
}

// HACK(eddyb) (see `ty::RegionKind::display_outputs_anything`)
//
// NB: this must be kept in sync with the printing logic above.
impl ty::RegionVid {
    fn display_outputs_anything<P>(&self, cx: &mut PrintCx<'_, '_, '_, P>) -> bool {
        if cx.is_verbose {
            return true;
        }

        let highlight = RegionHighlightMode::get();
        if highlight.region_highlighted(&ty::ReVar(*self)).is_some() {
            return true;
        }

        false
    }
}

define_print! {
    () ty::InferTy, (self, cx) {
        display {
            if cx.is_verbose {
                return self.print_debug(cx);
            }
            match *self {
                ty::TyVar(_) => p!(write("_")),
                ty::IntVar(_) => p!(write("{}", "{integer}")),
                ty::FloatVar(_) => p!(write("{}", "{float}")),
                ty::FreshTy(v) => p!(write("FreshTy({})", v)),
                ty::FreshIntTy(v) => p!(write("FreshIntTy({})", v)),
                ty::FreshFloatTy(v) => p!(write("FreshFloatTy({})", v))
            }
        }
        debug {
            match *self {
                ty::TyVar(ref v) => p!(write("{:?}", v)),
                ty::IntVar(ref v) => p!(write("{:?}", v)),
                ty::FloatVar(ref v) => p!(write("{:?}", v)),
                ty::FreshTy(v) => p!(write("FreshTy({:?})", v)),
                ty::FreshIntTy(v) => p!(write("FreshIntTy({:?})", v)),
                ty::FreshFloatTy(v) => p!(write("FreshFloatTy({:?})", v))
            }
        }
    }
}

impl fmt::Debug for ty::IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::IntType(ref v) => v.fmt(f),
            ty::UintType(ref v) => v.fmt(f),
        }
    }
}

impl fmt::Debug for ty::FloatVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// The generic impl doesn't work yet because projections are not
// normalized under HRTB.
/*impl<T> fmt::Display for ty::Binder<T>
    where T: fmt::Display + for<'a> ty::Lift<'a>,
          for<'a> <T as ty::Lift<'a>>::Lifted: fmt::Display + TypeFoldable<'a>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrintCx::with(|cx| cx.in_binder(cx.tcx.lift(self)
            .expect("could not lift for printing")))
    }
}*/

define_print_multi! {
    [
    ('tcx) ty::Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>,
    ('tcx) ty::Binder<ty::TraitRef<'tcx>>,
    ('tcx) ty::Binder<ty::FnSig<'tcx>>,
    ('tcx) ty::Binder<ty::TraitPredicate<'tcx>>,
    ('tcx) ty::Binder<ty::SubtypePredicate<'tcx>>,
    ('tcx) ty::Binder<ty::ProjectionPredicate<'tcx>>,
    ('tcx) ty::Binder<ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>,
    ('tcx) ty::Binder<ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>>
    ]
    (self, cx) {
        display {
            cx.in_binder(self)
        }
    }
}

define_print! {
    ('tcx) ty::TraitRef<'tcx>, (self, cx) {
        display {
            let _ = cx.print_def_path(
                self.def_id,
                Some(self.substs),
                Namespace::TypeNS,
                iter::empty(),
            )?;
            Ok(())
        }
        debug {
            let _ = cx.path_qualified(None, self.self_ty(), Some(*self), Namespace::TypeNS)?;
            Ok(())
        }
    }
}

define_print! {
    ('tcx) ty::Ty<'tcx>, (self, cx) {
        display {
            match self.sty {
                Bool => p!(write("bool")),
                Char => p!(write("char")),
                Int(t) => p!(write("{}", t.ty_to_string())),
                Uint(t) => p!(write("{}", t.ty_to_string())),
                Float(t) => p!(write("{}", t.ty_to_string())),
                RawPtr(ref tm) => {
                    p!(write("*{} ", match tm.mutbl {
                        hir::MutMutable => "mut",
                        hir::MutImmutable => "const",
                    }))?;
                    tm.ty.print(cx)
                }
                Ref(r, ty, mutbl) => {
                    p!(write("&"))?;
                    if r.display_outputs_anything(cx) {
                        p!(print_display(r), write(" "))?;
                    }
                    ty::TypeAndMut { ty, mutbl }.print(cx)
                }
                Never => p!(write("!")),
                Tuple(ref tys) => {
                    p!(write("("))?;
                    let mut tys = tys.iter();
                    if let Some(&ty) = tys.next() {
                        p!(print(ty), write(","))?;
                        if let Some(&ty) = tys.next() {
                            p!(write(" "), print(ty))?;
                            for &ty in tys {
                                p!(write(", "), print(ty))?;
                            }
                        }
                    }
                    p!(write(")"))
                }
                FnDef(def_id, substs) => {
                    let sig = cx.tcx.fn_sig(def_id).subst(cx.tcx, substs);
                    p!(print(sig), write(" {{"))?;
                    let _ = cx.print_def_path(
                        def_id,
                        Some(substs),
                        Namespace::ValueNS,
                        iter::empty(),
                    )?;
                    p!(write("}}"))
                }
                FnPtr(ref bare_fn) => {
                    bare_fn.print(cx)
                }
                Infer(infer_ty) => p!(write("{}", infer_ty)),
                Error => p!(write("[type error]")),
                Param(ref param_ty) => p!(write("{}", param_ty)),
                Bound(debruijn, bound_ty) => {
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
                Adt(def, substs) => {
                    let _ = cx.print_def_path(
                        def.did,
                        Some(substs),
                        Namespace::TypeNS,
                        iter::empty(),
                    )?;
                    Ok(())
                }
                Dynamic(data, r) => {
                    let print_r = r.display_outputs_anything(cx);
                    if print_r {
                        p!(write("("))?;
                    }
                    p!(write("dyn "))?;
                    data.print(cx)?;
                    if print_r {
                        p!(write(" + "), print_display(r), write(")"))?;
                    }
                    Ok(())
                }
                Foreign(def_id) => {
                    let _ = cx.print_def_path(
                        def_id,
                        None,
                        Namespace::TypeNS,
                        iter::empty(),
                    )?;
                    Ok(())
                }
                Projection(ref data) => data.print(cx),
                UnnormalizedProjection(ref data) => {
                    p!(write("Unnormalized("))?;
                    data.print(cx)?;
                    p!(write(")"))
                }
                Placeholder(placeholder) => {
                    p!(write("Placeholder({:?})", placeholder))
                }
                Opaque(def_id, substs) => {
                    if cx.is_verbose {
                        return p!(write("Opaque({:?}, {:?})", def_id, substs));
                    }

                    let def_key = cx.tcx.def_key(def_id);
                    if let Some(name) = def_key.disambiguated_data.data.get_opt_name() {
                        p!(write("{}", name))?;
                        let mut substs = substs.iter();
                        // FIXME(eddyb) print this with `print_def_path`.
                        if let Some(first) = substs.next() {
                            p!(write("::<"))?;
                            p!(write("{}", first))?;
                            for subst in substs {
                                p!(write(", {}", subst))?;
                            }
                            p!(write(">"))?;
                        }
                        return Ok(());
                    }
                    // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                    // by looking up the projections associated with the def_id.
                    let bounds = cx.tcx.predicates_of(def_id).instantiate(cx.tcx, substs);

                    let mut first = true;
                    let mut is_sized = false;
                    p!(write("impl"))?;
                    for predicate in bounds.predicates {
                        if let Some(trait_ref) = predicate.to_opt_poly_trait_ref() {
                            // Don't print +Sized, but rather +?Sized if absent.
                            if Some(trait_ref.def_id()) == cx.tcx.lang_items().sized_trait() {
                                is_sized = true;
                                continue;
                            }

                            p!(
                                    write("{}", if first { " " } else { "+" }),
                                    print(trait_ref))?;
                            first = false;
                        }
                    }
                    if !is_sized {
                        p!(write("{}?Sized", if first { " " } else { "+" }))?;
                    } else if first {
                        p!(write(" Sized"))?;
                    }
                    Ok(())
                }
                Str => p!(write("str")),
                Generator(did, substs, movability) => {
                    let upvar_tys = substs.upvar_tys(did, cx.tcx);
                    let witness = substs.witness(did, cx.tcx);
                    if movability == hir::GeneratorMovability::Movable {
                        p!(write("[generator"))?;
                    } else {
                        p!(write("[static generator"))?;
                    }

                    // FIXME(eddyb) should use `def_span`.
                    if let Some(hir_id) = cx.tcx.hir().as_local_hir_id(did) {
                        p!(write("@{:?}", cx.tcx.hir().span_by_hir_id(hir_id)))?;
                        let mut sep = " ";
                        cx.tcx.with_freevars(hir_id, |freevars| {
                            for (freevar, upvar_ty) in freevars.iter().zip(upvar_tys) {
                                p!(
                                       write("{}{}:",
                                             sep,
                                             cx.tcx.hir().name(freevar.var_id())),
                                       print(upvar_ty))?;
                                sep = ", ";
                            }
                            Ok(())
                        })?
                    } else {
                        // cross-crate closure types should only be
                        // visible in codegen bug reports, I imagine.
                        p!(write("@{:?}", did))?;
                        let mut sep = " ";
                        for (index, upvar_ty) in upvar_tys.enumerate() {
                            p!(
                                   write("{}{}:", sep, index),
                                   print(upvar_ty))?;
                            sep = ", ";
                        }
                    }

                    p!(write(" "), print(witness), write("]"))
                },
                GeneratorWitness(types) => {
                    cx.in_binder(&types)
                }
                Closure(did, substs) => {
                    let upvar_tys = substs.upvar_tys(did, cx.tcx);
                    p!(write("[closure"))?;

                    // FIXME(eddyb) should use `def_span`.
                    if let Some(hir_id) = cx.tcx.hir().as_local_hir_id(did) {
                        if cx.tcx.sess.opts.debugging_opts.span_free_formats {
                            p!(write("@{:?}", hir_id))?;
                        } else {
                            p!(write("@{:?}", cx.tcx.hir().span_by_hir_id(hir_id)))?;
                        }
                        let mut sep = " ";
                        cx.tcx.with_freevars(hir_id, |freevars| {
                            for (freevar, upvar_ty) in freevars.iter().zip(upvar_tys) {
                                p!(
                                       write("{}{}:",
                                             sep,
                                             cx.tcx.hir().name(freevar.var_id())),
                                       print(upvar_ty))?;
                                sep = ", ";
                            }
                            Ok(())
                        })?
                    } else {
                        // cross-crate closure types should only be
                        // visible in codegen bug reports, I imagine.
                        p!(write("@{:?}", did))?;
                        let mut sep = " ";
                        for (index, upvar_ty) in upvar_tys.enumerate() {
                            p!(
                                   write("{}{}:", sep, index),
                                   print(upvar_ty))?;
                            sep = ", ";
                        }
                    }

                    if cx.is_verbose {
                        p!(write(
                            " closure_kind_ty={:?} closure_sig_ty={:?}",
                            substs.closure_kind_ty(did, cx.tcx),
                            substs.closure_sig_ty(did, cx.tcx)
                        ))?;
                    }

                    p!(write("]"))
                },
                Array(ty, sz) => {
                    p!(write("["), print(ty), write("; "))?;
                    match sz {
                        ty::LazyConst::Unevaluated(_def_id, _substs) => {
                            p!(write("_"))?;
                        }
                        ty::LazyConst::Evaluated(c) => {
                            match c.val {
                                ConstValue::Infer(..) => p!(write("_"))?,
                                ConstValue::Param(ParamConst { name, .. }) =>
                                    p!(write("{}", name))?,
                                _ => p!(write("{}", c.unwrap_usize(cx.tcx)))?,
                            }
                        }
                    }
                    p!(write("]"))
                }
                Slice(ty) => {
                    p!(write("["), print(ty), write("]"))
                }
            }
        }
        debug {
            self.print_display(cx)
        }
    }
}

define_print! {
    ('tcx) ConstValue<'tcx>, (self, cx) {
        display {
            match self {
                ConstValue::Infer(..) => p!(write("_")),
                ConstValue::Param(ParamConst { name, .. }) => p!(write("{}", name)),
                _ => p!(write("{:?}", self)),
            }
        }
    }
}

define_print! {
    ('tcx) ty::Const<'tcx>, (self, cx) {
        display {
            p!(write("{} : {}", self.val, self.ty))
        }
    }
}

define_print! {
    ('tcx) ty::LazyConst<'tcx>, (self, cx) {
        display {
            match self {
                // FIXME(const_generics) this should print at least the type.
                ty::LazyConst::Unevaluated(..) => p!(write("_ : _")),
                ty::LazyConst::Evaluated(c) => p!(write("{}", c)),
            }
        }
    }
}

define_print! {
    () ty::ParamTy, (self, cx) {
        display {
            p!(write("{}", self.name))
        }
        debug {
            p!(write("{}/#{}", self.name, self.idx))
        }
    }
}

define_print! {
    () ty::ParamConst, (self, cx) {
        display {
            p!(write("{}", self.name))
        }
        debug {
            p!(write("{}/#{}", self.name, self.index))
        }
    }
}

// Similar problem to `Binder<T>`, can't define a generic impl.
define_print_multi! {
    [
    ('tcx) ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>,
    ('tcx) ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>
    ]
    (self, cx) {
        display {
            p!(print(self.0), write(" : "), print(self.1))
        }
    }
}

define_print! {
    ('tcx) ty::SubtypePredicate<'tcx>, (self, cx) {
        display {
            p!(print(self.a), write(" <: "), print(self.b))
        }
    }
}

define_print! {
    ('tcx) ty::TraitPredicate<'tcx>, (self, cx) {
        debug {
            p!(write("TraitPredicate({:?})",
                   self.trait_ref))
        }
        display {
            p!(print(self.trait_ref.self_ty()), write(": "), print(self.trait_ref))
        }
    }
}

define_print! {
    ('tcx) ty::ProjectionPredicate<'tcx>, (self, cx) {
        debug {
            p!(
                   write("ProjectionPredicate("),
                   print(self.projection_ty),
                   write(", "),
                   print(self.ty),
                   write(")"))
        }
        display {
            p!(print(self.projection_ty), write(" == "), print(self.ty))
        }
    }
}

define_print! {
    ('tcx) ty::ProjectionTy<'tcx>, (self, cx) {
        display {
            let _ = cx.print_def_path(
                self.item_def_id,
                Some(self.substs),
                Namespace::TypeNS,
                iter::empty(),
            )?;
            Ok(())
        }
    }
}

define_print! {
    () ty::ClosureKind, (self, cx) {
        display {
            match *self {
                ty::ClosureKind::Fn => p!(write("Fn")),
                ty::ClosureKind::FnMut => p!(write("FnMut")),
                ty::ClosureKind::FnOnce => p!(write("FnOnce")),
            }
        }
    }
}

define_print! {
    ('tcx) ty::Predicate<'tcx>, (self, cx) {
        display {
            match *self {
                ty::Predicate::Trait(ref data) => data.print(cx),
                ty::Predicate::Subtype(ref predicate) => predicate.print(cx),
                ty::Predicate::RegionOutlives(ref predicate) => predicate.print(cx),
                ty::Predicate::TypeOutlives(ref predicate) => predicate.print(cx),
                ty::Predicate::Projection(ref predicate) => predicate.print(cx),
                ty::Predicate::WellFormed(ty) => p!(print(ty), write(" well-formed")),
                ty::Predicate::ObjectSafe(trait_def_id) => {
                    p!(write("the trait `"))?;
                    let _ = cx.print_def_path(
                        trait_def_id,
                        None,
                        Namespace::TypeNS,
                        iter::empty(),
                    )?;
                    p!(write("` is object-safe"))
                }
                ty::Predicate::ClosureKind(closure_def_id, _closure_substs, kind) => {
                    p!(write("the closure `"))?;
                    let _ = cx.print_def_path(
                        closure_def_id,
                        None,
                        Namespace::ValueNS,
                        iter::empty(),
                    )?;
                    p!(write("` implements the trait `{}`", kind))
                }
                ty::Predicate::ConstEvaluatable(def_id, substs) => {
                    p!(write("the constant `"))?;
                    let _ = cx.print_def_path(
                        def_id,
                        Some(substs),
                        Namespace::ValueNS,
                        iter::empty(),
                    )?;
                    p!(write("` can be evaluated"))
                }
            }
        }
        debug {
            match *self {
                ty::Predicate::Trait(ref a) => a.print(cx),
                ty::Predicate::Subtype(ref pair) => pair.print(cx),
                ty::Predicate::RegionOutlives(ref pair) => pair.print(cx),
                ty::Predicate::TypeOutlives(ref pair) => pair.print(cx),
                ty::Predicate::Projection(ref pair) => pair.print(cx),
                ty::Predicate::WellFormed(ty) => ty.print(cx),
                ty::Predicate::ObjectSafe(trait_def_id) => {
                    p!(write("ObjectSafe({:?})", trait_def_id))
                }
                ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                    p!(write("ClosureKind({:?}, {:?}, {:?})",
                        closure_def_id, closure_substs, kind))
                }
                ty::Predicate::ConstEvaluatable(def_id, substs) => {
                    p!(write("ConstEvaluatable({:?}, {:?})", def_id, substs))
                }
            }
        }
    }
}

define_print! {
    ('tcx) Kind<'tcx>, (self, cx) {
        display {
            match self.unpack() {
                UnpackedKind::Lifetime(lt) => p!(print(lt)),
                UnpackedKind::Type(ty) => p!(print(ty)),
                UnpackedKind::Const(ct) => p!(print(ct)),
            }
        }
        debug {
            match self.unpack() {
                UnpackedKind::Lifetime(lt) => p!(print(lt)),
                UnpackedKind::Type(ty) => p!(print(ty)),
                UnpackedKind::Const(ct) => p!(print(ct)),
            }
        }
    }
}
