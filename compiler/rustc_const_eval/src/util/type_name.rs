use std::fmt::Write;

use rustc_data_structures::intern::Interned;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::DisambiguatedDefPathData;
use rustc_middle::bug;
use rustc_middle::ty::print::{PrettyPrinter, PrintError, Printer};
use rustc_middle::ty::{self, GenericArg, Ty, TyCtxt};

struct TypeNamePrinter<'tcx> {
    tcx: TyCtxt<'tcx>,
    path: String,
}

impl<'tcx> Printer<'tcx> for TypeNamePrinter<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_region(&mut self, _region: ty::Region<'_>) -> Result<(), PrintError> {
        // FIXME: most regions have been erased by the time this code runs.
        // Just printing `'_` is a bit hacky but gives mostly good results, and
        // doing better is difficult. See `should_print_optional_region`.
        write!(self, "'_")
    }

    fn print_type(&mut self, ty: Ty<'tcx>) -> Result<(), PrintError> {
        match *ty.kind() {
            // Types without identity.
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Pat(_, _)
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnPtr(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Dynamic(_, _)
            | ty::UnsafeBinder(_) => self.pretty_print_type(ty),

            // Placeholders (all printed as `_` to uniformize them).
            ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => {
                write!(self, "_")?;
                Ok(())
            }

            // Types with identity (print the module path).
            ty::Adt(ty::AdtDef(Interned(&ty::AdtDefData { did: def_id, .. }, _)), args)
            | ty::FnDef(def_id, args)
            | ty::Alias(ty::Projection | ty::Opaque, ty::AliasTy { def_id, args, .. })
            | ty::Closure(def_id, args)
            | ty::CoroutineClosure(def_id, args)
            | ty::Coroutine(def_id, args) => self.print_def_path(def_id, args),
            ty::Foreign(def_id) => self.print_def_path(def_id, &[]),

            ty::Alias(ty::Free, _) => bug!("type_name: unexpected free alias"),
            ty::Alias(ty::Inherent, _) => bug!("type_name: unexpected inherent projection"),
            ty::CoroutineWitness(..) => bug!("type_name: unexpected `CoroutineWitness`"),
        }
    }

    fn print_const(&mut self, ct: ty::Const<'tcx>) -> Result<(), PrintError> {
        self.pretty_print_const(ct, false)
    }

    fn print_dyn_existential(
        &mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<(), PrintError> {
        self.pretty_print_dyn_existential(predicates)
    }

    fn print_crate_name(&mut self, cnum: CrateNum) -> Result<(), PrintError> {
        self.path.push_str(self.tcx.crate_name(cnum).as_str());
        Ok(())
    }

    fn print_path_with_qualified(
        &mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        self.pretty_print_path_with_qualified(self_ty, trait_ref)
    }

    fn print_path_with_impl(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        self.pretty_print_path_with_impl(
            |cx| {
                print_prefix(cx)?;

                cx.path.push_str("::");

                Ok(())
            },
            self_ty,
            trait_ref,
        )
    }

    fn print_path_with_simple(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<(), PrintError> {
        print_prefix(self)?;

        write!(self.path, "::{}", disambiguated_data.data).unwrap();

        Ok(())
    }

    fn print_path_with_generic_args(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        print_prefix(self)?;
        if !args.is_empty() {
            self.generic_delimiters(|cx| cx.comma_sep(args.iter().copied()))
        } else {
            Ok(())
        }
    }

    fn print_coroutine_with_kind(
        &mut self,
        def_id: DefId,
        parent_args: &'tcx [GenericArg<'tcx>],
        kind: Ty<'tcx>,
    ) -> Result<(), PrintError> {
        self.print_def_path(def_id, parent_args)?;

        let ty::Coroutine(_, args) = self.tcx.type_of(def_id).instantiate_identity().kind() else {
            // Could be `ty::Error`.
            return Ok(());
        };

        let default_kind = args.as_coroutine().kind_ty();

        match kind.to_opt_closure_kind() {
            _ if kind == default_kind => {
                // No need to mark the closure if it's the deduced coroutine kind.
            }
            Some(ty::ClosureKind::Fn) | None => {
                // Should never happen. Just don't mark anything rather than panicking.
            }
            Some(ty::ClosureKind::FnMut) => self.path.push_str("::{{call_mut}}"),
            Some(ty::ClosureKind::FnOnce) => self.path.push_str("::{{call_once}}"),
        }

        Ok(())
    }
}

impl<'tcx> PrettyPrinter<'tcx> for TypeNamePrinter<'tcx> {
    fn should_print_optional_region(&self, region: ty::Region<'_>) -> bool {
        // Bound regions are always printed (as `'_`), which gives some idea that they are special,
        // even though the `for` is omitted by the pretty printer.
        // E.g. `for<'a, 'b> fn(&'a u32, &'b u32)` is printed as "fn(&'_ u32, &'_ u32)".
        let kind = region.kind();
        match region.kind() {
            ty::ReErased | ty::ReEarlyParam(_) | ty::ReStatic => false,
            ty::ReBound(..) => true,
            _ => panic!("type_name unhandled region: {kind:?}"),
        }
    }

    fn generic_delimiters(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<(), PrintError>,
    ) -> Result<(), PrintError> {
        write!(self, "<")?;

        f(self)?;

        write!(self, ">")?;

        Ok(())
    }

    fn should_print_verbose(&self) -> bool {
        // `std::any::type_name` should never print verbose type names
        false
    }
}

impl Write for TypeNamePrinter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.path.push_str(s);
        Ok(())
    }
}

pub fn type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> String {
    let mut p = TypeNamePrinter { tcx, path: String::new() };
    p.print_type(ty).unwrap();
    p.path
}
