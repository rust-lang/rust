use rustc_hir::def_id::CrateNum;
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::mir::interpret::Allocation;
use rustc_middle::ty::{
    self,
    print::{PrettyPrinter, Print, Printer},
    subst::{GenericArg, GenericArgKind},
    Ty, TyCtxt,
};
use std::fmt::Write;

struct AbsolutePathPrinter<'tcx> {
    tcx: TyCtxt<'tcx>,
    path: String,
}

impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
    type Error = std::fmt::Error;

    type Path = Self;
    type Region = Self;
    type Type = Self;
    type DynExistential = Self;
    type Const = Self;

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_region(self, _region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
        Ok(self)
    }

    fn print_type(mut self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
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
            | ty::Dynamic(_, _) => self.pretty_print_type(ty),

            // Placeholders (all printed as `_` to uniformize them).
            ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => {
                write!(self, "_")?;
                Ok(self)
            }

            // Types with identity (print the module path).
            ty::Adt(&ty::AdtDef { did: def_id, .. }, substs)
            | ty::FnDef(def_id, substs)
            | ty::Opaque(def_id, substs)
            | ty::Projection(ty::ProjectionTy { item_def_id: def_id, substs })
            | ty::Closure(def_id, substs)
            | ty::Generator(def_id, substs, _) => self.print_def_path(def_id, substs),
            ty::Foreign(def_id) => self.print_def_path(def_id, &[]),

            ty::GeneratorWitness(_) => bug!("type_name: unexpected `GeneratorWitness`"),
        }
    }

    fn print_const(self, ct: &'tcx ty::Const<'tcx>) -> Result<Self::Const, Self::Error> {
        self.pretty_print_const(ct, false)
    }

    fn print_dyn_existential(
        mut self,
        predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        let mut first = true;
        for p in predicates {
            if !first {
                write!(self, "+")?;
            }
            first = false;
            self = p.print(self)?;
        }
        Ok(self)
    }

    fn path_crate(mut self, cnum: CrateNum) -> Result<Self::Path, Self::Error> {
        self.path.push_str(&self.tcx.original_crate_name(cnum).as_str());
        Ok(self)
    }

    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.pretty_path_qualified(self_ty, trait_ref)
    }

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        _disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
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
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;

        // Skip `::{{constructor}}` on tuple/unit structs.
        if disambiguated_data.data == DefPathData::Ctor {
            return Ok(self);
        }

        self.path.push_str("::");

        self.path.push_str(&disambiguated_data.data.as_symbol().as_str());
        Ok(self)
    }

    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;
        let args = args.iter().cloned().filter(|arg| match arg.unpack() {
            GenericArgKind::Lifetime(_) => false,
            _ => true,
        });
        if args.clone().next().is_some() {
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(self)
        }
    }
}

impl PrettyPrinter<'tcx> for AbsolutePathPrinter<'tcx> {
    fn region_should_not_be_omitted(&self, _region: ty::Region<'_>) -> bool {
        false
    }
    fn comma_sep<T>(mut self, mut elems: impl Iterator<Item = T>) -> Result<Self, Self::Error>
    where
        T: Print<'tcx, Self, Output = Self, Error = Self::Error>,
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
        f: impl FnOnce(Self) -> Result<Self, Self::Error>,
    ) -> Result<Self, Self::Error> {
        write!(self, "<")?;

        self = f(self)?;

        write!(self, ">")?;

        Ok(self)
    }
}

impl Write for AbsolutePathPrinter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.path.push_str(s);
        Ok(())
    }
}

/// Directly returns an `Allocation` containing an absolute path representation of the given type.
crate fn alloc_type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> &'tcx Allocation {
    let path = AbsolutePathPrinter { tcx, path: String::new() }.print_type(ty).unwrap().path;
    let alloc = Allocation::from_byte_aligned_bytes(path.into_bytes());
    tcx.intern_const_alloc(alloc)
}
