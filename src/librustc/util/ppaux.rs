use crate::hir;
use crate::hir::def::Namespace;
use crate::ty::subst::{Kind, UnpackedKind};
use crate::ty::{self, ParamConst, Ty};
use crate::ty::print::{FmtPrinter, PrettyPrinter, PrintCx, Print};
use crate::mir::interpret::ConstValue;

use std::fmt;
use std::iter;

use rustc_target::spec::abi::Abi;

macro_rules! define_print {
    ([$($target:ty),+] $vars:tt $def:tt) => {
        $(define_print!($target, $vars $def);)+
    };

    ($target:ty, ($self:ident, $cx:ident) { display $disp:block }) => {
        impl<P: PrettyPrinter> Print<'tcx, P> for $target {
            type Output = P;
            type Error = fmt::Error;
            fn print(&$self, $cx: PrintCx<'_, '_, 'tcx, P>) -> Result<Self::Output, Self::Error> {
                #[allow(unused_mut)]
                let mut $cx = $cx;
                define_scoped_cx!($cx);
                let _: () = $disp;
                #[allow(unreachable_code)]
                Ok($cx.printer)
            }
        }

        impl fmt::Display for $target {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                PrintCx::with_tls_tcx(FmtPrinter::new(f, Namespace::TypeNS), |cx| {
                    cx.tcx.lift(self).expect("could not lift for printing").print(cx)?;
                    Ok(())
                })
            }
        }
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

define_print! {
    &'tcx ty::List<ty::ExistentialPredicate<'tcx>>, (self, cx) {
        display {
            // Generate the main trait ref, including associated types.
            let mut first = true;

            if let Some(principal) = self.principal() {
                let mut resugared_principal = false;

                // Special-case `Fn(...) -> ...` and resugar it.
                let fn_trait_kind = cx.tcx.lang_items().fn_trait_kind(principal.def_id);
                if !cx.tcx.sess.verbose() && fn_trait_kind.is_some() {
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

define_print! {
    &'tcx ty::List<Ty<'tcx>>, (self, cx) {
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
    ty::TypeAndMut<'tcx>, (self, cx) {
        display {
            p!(
                   write("{}", if self.mutbl == hir::MutMutable { "mut " } else { "" }),
                   print(self.ty))
        }
    }
}

define_print! {
    ty::ExistentialTraitRef<'tcx>, (self, cx) {
        display {
            let dummy_self = cx.tcx.mk_infer(ty::FreshTy(0));

            let trait_ref = *ty::Binder::bind(*self)
                .with_self_ty(cx.tcx, dummy_self)
                .skip_binder();
            p!(print(trait_ref))
        }
    }
}

define_print! {
    ty::RegionKind, (self, cx) {
        display {
            return cx.print_region(self);
        }
    }
}

define_print! {
    ty::FnSig<'tcx>, (self, cx) {
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
    }
}

define_print! {
    ty::InferTy, (self, cx) {
        display {
            if cx.tcx.sess.verbose() {
                p!(write("{:?}", self));
                return Ok(cx.printer);
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

define_print! {
    [
        ty::Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>,
        ty::Binder<ty::TraitRef<'tcx>>,
        ty::Binder<ty::FnSig<'tcx>>,
        ty::Binder<ty::TraitPredicate<'tcx>>,
        ty::Binder<ty::SubtypePredicate<'tcx>>,
        ty::Binder<ty::ProjectionPredicate<'tcx>>,
        ty::Binder<ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>>,
        ty::Binder<ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>>
    ]
    (self, cx) {
        display {
            nest!(|cx| cx.pretty_in_binder(self))
        }
    }
}

define_print! {
    ty::TraitRef<'tcx>, (self, cx) {
        display {
            nest!(|cx| cx.print_def_path(self.def_id, Some(self.substs), iter::empty()));
        }
    }
}

define_print! {
    Ty<'tcx>, (self, cx) {
        display {
            return cx.print_type(self);
        }
    }
}

define_print! {
    ConstValue<'tcx>, (self, cx) {
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
    ty::Const<'tcx>, (self, cx) {
        display {
            p!(write("{} : {}", self.val, self.ty))
        }
    }
}

define_print! {
    ty::LazyConst<'tcx>, (self, cx) {
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
    ty::ParamTy, (self, cx) {
        display {
            p!(write("{}", self.name))
        }
    }
}

define_print! {
    ty::ParamConst, (self, cx) {
        display {
            p!(write("{}", self.name))
        }
    }
}

// Similar problem to `Binder<T>`, can't define a generic impl.
define_print! {
    [
        ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>,
        ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>
    ]
    (self, cx) {
        display {
            p!(print(self.0), write(" : "), print(self.1))
        }
    }
}

define_print! {
    ty::SubtypePredicate<'tcx>, (self, cx) {
        display {
            p!(print(self.a), write(" <: "), print(self.b))
        }
    }
}

define_print! {
    ty::TraitPredicate<'tcx>, (self, cx) {
        display {
            p!(print(self.trait_ref.self_ty()), write(": "), print(self.trait_ref))
        }
    }
}

define_print! {
    ty::ProjectionPredicate<'tcx>, (self, cx) {
        display {
            p!(print(self.projection_ty), write(" == "), print(self.ty))
        }
    }
}

define_print! {
    ty::ProjectionTy<'tcx>, (self, cx) {
        display {
            nest!(|cx| cx.print_def_path(self.item_def_id, Some(self.substs), iter::empty()));
        }
    }
}

define_print! {
    ty::ClosureKind, (self, cx) {
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
    ty::Predicate<'tcx>, (self, cx) {
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
    }
}

define_print! {
    Kind<'tcx>, (self, cx) {
        display {
            match self.unpack() {
                UnpackedKind::Lifetime(lt) => p!(print(lt)),
                UnpackedKind::Type(ty) => p!(print(ty)),
                UnpackedKind::Const(ct) => p!(print(ct)),
            }
        }
    }
}
