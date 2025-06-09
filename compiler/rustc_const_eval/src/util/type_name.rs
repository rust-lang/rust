use std::fmt::Write;

use rustc_data_structures::intern::Interned;
use rustc_hir::def_id::CrateNum;
use rustc_hir::definitions::DisambiguatedDefPathData;
use rustc_middle::bug;
use rustc_middle::ty::print::{PrettyPrinter, Print, PrintError, Printer};
use rustc_middle::ty::{self, GenericArg, GenericArgKind, Ty, TyCtxt};

struct AbsolutePathPrinter<'tcx> {
    tcx: TyCtxt<'tcx>,
    path: String,
}

impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_region(&mut self, _region: ty::Region<'_>) -> Result<(), PrintError> {
        Ok(())
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

    fn path_crate(&mut self, cnum: CrateNum) -> Result<(), PrintError> {
        self.path.push_str(self.tcx.crate_name(cnum).as_str());
        Ok(())
    }

    fn path_qualified(
        &mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        self.pretty_path_qualified(self_ty, trait_ref)
    }

    fn path_append_impl(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        _disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        self.pretty_path_append_impl(
            |cx| {
                print_prefix(cx)?;

                cx.path.push_str("::");

                Ok(())
            },
            self_ty,
            trait_ref,
        )
    }

    fn path_append(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<(), PrintError> {
        print_prefix(self)?;

        write!(self.path, "::{}", disambiguated_data.data).unwrap();

        Ok(())
    }

    fn path_generic_args(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        print_prefix(self)?;
        let args =
            args.iter().cloned().filter(|arg| !matches!(arg.kind(), GenericArgKind::Lifetime(_)));
        if args.clone().next().is_some() {
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(())
        }
    }
}

impl<'tcx> PrettyPrinter<'tcx> for AbsolutePathPrinter<'tcx> {
    fn should_print_region(&self, _region: ty::Region<'_>) -> bool {
        false
    }
    fn comma_sep<T>(&mut self, mut elems: impl Iterator<Item = T>) -> Result<(), PrintError>
    where
        T: Print<'tcx, Self>,
    {
        if let Some(first) = elems.next() {
            first.print(self)?;
            for elem in elems {
                self.path.push_str(", ");
                elem.print(self)?;
            }
        }
        Ok(())
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

impl Write for AbsolutePathPrinter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.path.push_str(s);
        Ok(())
    }
}

pub fn type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> String {
    let mut printer = AbsolutePathPrinter { tcx, path: String::new() };
    printer.print_type(ty).unwrap();
    printer.path
}
