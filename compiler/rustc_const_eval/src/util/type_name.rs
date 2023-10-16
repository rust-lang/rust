use rustc_data_structures::intern::Interned;
use rustc_hir::def_id::CrateNum;
use rustc_hir::definitions::DisambiguatedDefPathData;
use rustc_middle::ty::{
    self,
    print::{PrettyPrinter, Print, PrintError, Printer},
    GenericArg, GenericArgKind, Ty, TyCtxt,
};
use std::fmt::Write;

struct AbsolutePathPrinter<'tcx> {
    tcx: TyCtxt<'tcx>,
    path: String,
}

impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_region(self, _region: ty::Region<'_>) -> Result<Self, PrintError> {
        Ok(self)
    }

    fn print_type(mut self, ty: Ty<'tcx>) -> Result<Self, PrintError> {
        match *ty.kind() {
            // Types without identity.
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnPtr(_)
            | ty::Never
            | ty::Tuple(_)
            | ty::Dynamic(_, _, _) => self.pretty_print_type(ty),

            // Placeholders (all printed as `_` to uniformize them).
            ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => {
                write!(self, "_")?;
                Ok(self)
            }

            // Types with identity (print the module path).
            ty::Adt(ty::AdtDef(Interned(&ty::AdtDefData { did: def_id, .. }, _)), args)
            | ty::FnDef(def_id, args)
            | ty::Alias(ty::Projection | ty::Opaque, ty::AliasTy { def_id, args, .. })
            | ty::Closure(def_id, args)
            | ty::Generator(def_id, args, _) => self.print_def_path(def_id, args),
            ty::Foreign(def_id) => self.print_def_path(def_id, &[]),

            ty::Alias(ty::Weak, _) => bug!("type_name: unexpected weak projection"),
            ty::Alias(ty::Inherent, _) => bug!("type_name: unexpected inherent projection"),
            ty::GeneratorWitness(..) => bug!("type_name: unexpected `GeneratorWitness`"),
        }
    }

    fn print_const(self, ct: ty::Const<'tcx>) -> Result<Self, PrintError> {
        self.pretty_print_const(ct, false)
    }

    fn print_dyn_existential(
        self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<Self, PrintError> {
        self.pretty_print_dyn_existential(predicates)
    }

    fn path_crate(mut self, cnum: CrateNum) -> Result<Self, PrintError> {
        self.path.push_str(self.tcx.crate_name(cnum).as_str());
        Ok(self)
    }

    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self, PrintError> {
        self.pretty_path_qualified(self_ty, trait_ref)
    }

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self, PrintError>,
        _disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self, PrintError> {
        self.pretty_path_append_impl(
            |mut cx| {
                cx = print_prefix(cx)?;

                cx.path.push_str("::");

                Ok(cx)
            },
            self_ty,
            trait_ref,
        )
    }

    fn path_append(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self, PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self, PrintError> {
        self = print_prefix(self)?;

        write!(self.path, "::{}", disambiguated_data.data).unwrap();

        Ok(self)
    }

    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self, PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<Self, PrintError> {
        self = print_prefix(self)?;
        let args =
            args.iter().cloned().filter(|arg| !matches!(arg.unpack(), GenericArgKind::Lifetime(_)));
        if args.clone().next().is_some() {
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(self)
        }
    }
}

impl<'tcx> PrettyPrinter<'tcx> for AbsolutePathPrinter<'tcx> {
    fn should_print_region(&self, _region: ty::Region<'_>) -> bool {
        false
    }
    fn comma_sep<T>(mut self, mut elems: impl Iterator<Item = T>) -> Result<Self, PrintError>
    where
        T: Print<'tcx, Self>,
    {
        if let Some(first) = elems.next() {
            self = first.print(self)?;
            for elem in elems {
                self.path.push_str(", ");
                self = elem.print(self)?;
            }
        }
        Ok(self)
    }

    fn generic_delimiters(
        mut self,
        f: impl FnOnce(Self) -> Result<Self, PrintError>,
    ) -> Result<Self, PrintError> {
        write!(self, "<")?;

        self = f(self)?;

        write!(self, ">")?;

        Ok(self)
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
    AbsolutePathPrinter { tcx, path: String::new() }.print_type(ty).unwrap().path
}
