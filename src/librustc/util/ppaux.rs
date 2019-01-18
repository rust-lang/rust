use crate::hir;
use crate::hir::def::Namespace;
use crate::hir::def_id::DefId;
use crate::ty::subst::{Kind, SubstsRef, UnpackedKind};
use crate::ty::{self, ParamConst, Ty};
use crate::ty::print::{FmtPrinter, PrettyPrinter, PrintCx, Print, Printer};
use crate::mir::interpret::ConstValue;

use std::fmt::{self, Write as _};
use std::iter;

use rustc_target::spec::abi::Abi;

macro_rules! gen_display_debug_body {
    ( $with:path ) => {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            PrintCx::with_tls_tcx(FmtPrinter::new(f, Namespace::TypeNS), |cx| {
                $with(&cx.tcx.lift(self).expect("could not lift for printing"), cx)?;
                Ok(())
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
            type Output = P;
            type Error = fmt::Error;
            fn print(&$self, $cx: PrintCx<'_, '_, 'tcx, P>) -> Result<Self::Output, Self::Error> {
                #[allow(unused_mut)]
                let mut $cx = $cx;
                let _: () = {
                    define_scoped_cx!($cx);

                    if $cx.config.is_debug $dbg
                    else $disp
                };
                Ok($cx.printer)
            }
        }
    };
    ( () $target:ty, ($self:ident, $cx:ident) $disp:block $dbg:block ) => {
        impl<P: PrettyPrinter> Print<'tcx, P> for $target {
            type Output = P;
            type Error = fmt::Error;
            fn print(&$self, $cx: PrintCx<'_, '_, 'tcx, P>) -> Result<Self::Output, Self::Error> {
                #[allow(unused_mut)]
                let mut $cx = $cx;
                let _: () = {
                    define_scoped_cx!($cx);

                    if $cx.config.is_debug $dbg
                    else $disp
                };
                Ok($cx.printer)
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
            write!($cx.printer, "{:?}", $self)?
        } }
    };
}
macro_rules! define_print_multi {
    ( [ $($generic:tt $target:ty),* ] $vars:tt $def:tt ) => {
        $(define_print! { $generic $target, $vars $def })*
    };
}
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

pub fn parameterized<F: fmt::Write>(
    f: &mut F,
    did: DefId,
    substs: SubstsRef<'_>,
    ns: Namespace,
) -> fmt::Result {
    PrintCx::with_tls_tcx(FmtPrinter::new(f, ns), |cx| {
        let substs = cx.tcx.lift(&substs).expect("could not lift for printing");
        cx.print_def_path(did, Some(substs), iter::empty())?;
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
                let fn_trait_kind = cx.tcx.lang_items().fn_trait_kind(principal.def_id);
                if !cx.config.is_verbose && fn_trait_kind.is_some() {
                    if let ty::Tuple(ref args) = principal.substs.type_at(0).sty {
                        let mut projections = self.projection_bounds();
                        if let (Some(proj), None) = (projections.next(), projections.next()) {
                            nest!(|cx| cx.print_def_path(principal.def_id, None, iter::empty()));
                            nest!(|cx| cx.pretty_fn_sig(args, false, proj.ty));
                            resugared_principal = true;
                        }
                    }
                }

                if !resugared_principal {
                    // Use a type that can't appear in defaults of type parameters.
                    let dummy_self = cx.tcx.mk_infer(ty::FreshTy(0));
                    let principal = principal.with_self_ty(cx.tcx, dummy_self);
                    nest!(|cx| cx.print_def_path(
                        principal.def_id,
                        Some(principal.substs),
                        self.projection_bounds(),
                    ));
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
                    p!(write(" + "));
                }
                first = false;

                nest!(|cx| cx.print_def_path(def_id, None, iter::empty()));
            }
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
        PrintCx::with_tls_tcx(FmtPrinter::new(f, Namespace::TypeNS), |cx| {
            cx.print_def_path(self.def_id, None, iter::empty())?;
            Ok(())
        })
    }
}

impl fmt::Debug for ty::AdtDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrintCx::with_tls_tcx(FmtPrinter::new(f, Namespace::TypeNS), |cx| {
            cx.print_def_path(self.did, None, iter::empty())?;
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
        PrintCx::with_tls_tcx(FmtPrinter::new(f, Namespace::ValueNS), |mut cx| {
            define_scoped_cx!(cx);
            p!(write("UpvarId({:?};`{}`;{:?})",
                self.var_path.hir_id,
                cx.tcx.hir().name_by_hir_id(self.var_path.hir_id),
                self.closure_expr_id));
            Ok(())
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
            p!(write("{{"));
            let mut tys = self.iter();
            if let Some(&ty) = tys.next() {
                p!(print(ty));
                for &ty in tys {
                    p!(write(", "), print(ty));
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

            let trait_ref = *ty::Binder::bind(*self)
                .with_self_ty(cx.tcx, dummy_self)
                .skip_binder();
            p!(print_display(trait_ref))
        }
        debug {
            p!(print_display(self))
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

impl fmt::Debug for ty::BoundRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::BrAnon(n) => write!(f, "BrAnon({:?})", n),
            ty::BrFresh(n) => write!(f, "BrFresh({:?})", n),
            ty::BrNamed(did, name) => {
                write!(f, "BrNamed({:?}:{:?}, {})",
                        did.krate, did.index, name)
            }
            ty::BrEnv => write!(f, "BrEnv"),
        }
    }
}

define_print! {
    () ty::RegionKind, (self, cx) {
        display {
            return cx.print_region(self);
        }
        debug {
            match *self {
                ty::ReEarlyBound(ref data) => {
                    p!(write("ReEarlyBound({}, {})",
                           data.index,
                           data.name))
                }

                ty::ReClosureBound(ref vid) => {
                    p!(write("ReClosureBound({:?})", vid))
                }

                ty::ReLateBound(binder_id, ref bound_region) => {
                    p!(write("ReLateBound({:?}, {:?})", binder_id, bound_region))
                }

                ty::ReFree(ref fr) => p!(print_debug(fr)),

                ty::ReScope(id) => {
                    p!(write("ReScope({:?})", id))
                }

                ty::ReStatic => p!(write("ReStatic")),

                ty::ReVar(ref vid) => {
                    p!(write("{:?}", vid));
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
            })?
        }
    }
}

define_print! {
    ('tcx) ty::FnSig<'tcx>, (self, cx) {
        display {
            if self.unsafety == hir::Unsafety::Unsafe {
                p!(write("unsafe "));
            }

            if self.abi != Abi::Rust {
                p!(write("extern {} ", self.abi));
            }

            p!(write("fn"));
            nest!(|cx| cx.pretty_fn_sig(self.inputs(), self.c_variadic, self.output()));
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

impl fmt::Debug for ty::RegionVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'_#{}r", self.index())
    }
}

define_print! {
    () ty::InferTy, (self, cx) {
        display {
            if cx.config.is_verbose {
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
        PrintCx::with_tls_tcx(|cx| cx.pretty_in_binder(cx.tcx.lift(self)
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
            nest!(|cx| cx.pretty_in_binder(self))
        }
    }
}

define_print! {
    ('tcx) ty::TraitRef<'tcx>, (self, cx) {
        display {
            nest!(|cx| cx.print_def_path(self.def_id, Some(self.substs), iter::empty()));
        }
        debug {
            nest!(|cx| cx.path_qualified(self.self_ty(), Some(*self)));
        }
    }
}

define_print! {
    ('tcx) Ty<'tcx>, (self, cx) {
        display {
            return cx.print_type(self);
        }
        debug {
            p!(print_display(self))
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
            nest!(|cx| cx.print_def_path(self.item_def_id, Some(self.substs), iter::empty()));
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
                ty::Predicate::Trait(ref data) => p!(print(data)),
                ty::Predicate::Subtype(ref predicate) => p!(print(predicate)),
                ty::Predicate::RegionOutlives(ref predicate) => p!(print(predicate)),
                ty::Predicate::TypeOutlives(ref predicate) => p!(print(predicate)),
                ty::Predicate::Projection(ref predicate) => p!(print(predicate)),
                ty::Predicate::WellFormed(ty) => p!(print(ty), write(" well-formed")),
                ty::Predicate::ObjectSafe(trait_def_id) => {
                    p!(write("the trait `"));
                    nest!(|cx| cx.print_def_path(trait_def_id, None, iter::empty()));
                    p!(write("` is object-safe"))
                }
                ty::Predicate::ClosureKind(closure_def_id, _closure_substs, kind) => {
                    p!(write("the closure `"));
                    nest!(|cx| cx.print_value_path(closure_def_id, None));
                    p!(write("` implements the trait `{}`", kind))
                }
                ty::Predicate::ConstEvaluatable(def_id, substs) => {
                    p!(write("the constant `"));
                    nest!(|cx| cx.print_value_path(def_id, Some(substs)));
                    p!(write("` can be evaluated"))
                }
            }
        }
        debug {
            match *self {
                ty::Predicate::Trait(ref a) => p!(print(a)),
                ty::Predicate::Subtype(ref pair) => p!(print(pair)),
                ty::Predicate::RegionOutlives(ref pair) => p!(print(pair)),
                ty::Predicate::TypeOutlives(ref pair) => p!(print(pair)),
                ty::Predicate::Projection(ref pair) => p!(print(pair)),
                ty::Predicate::WellFormed(ty) => p!(print(ty)),
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
