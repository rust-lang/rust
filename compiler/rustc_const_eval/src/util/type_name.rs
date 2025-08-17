use std::fmt::Write;

use rustc_data_structures::intern::Interned;
use rustc_hir::def_id::CrateNum;
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
            | ty::Dynamic(_, _, _)
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
}

impl<'tcx> PrettyPrinter<'tcx> for TypeNamePrinter<'tcx> {
    fn should_print_optional_region(&self, _region: ty::Region<'_>) -> bool {
        // Bound regions are always printed (as `'_`), which gives some idea that they are special,
        // even though the `for` is omitted by the pretty printer.
        // E.g. `for<'a, 'b> fn(&'a u32, &'b u32)` is printed as "fn(&'_ u32, &'_ u32)".
        match _region.kind() {
            ty::ReErased => false,
            ty::ReBound(..) => true,
            _ => unreachable!(),
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
