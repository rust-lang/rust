use rustc::ty::{
    TyCtxt, Ty,
    subst::{UnpackedKind, Kind},
    print::{Printer, PrettyPrinter, Print},
    self,
};
use rustc::hir::map::{DefPathData, DisambiguatedDefPathData};
use rustc::hir::def_id::CrateNum;
use std::fmt::Write;
use rustc::mir::interpret::{Allocation, ConstValue};

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
        match ty.sty {
            // Types without identity.
            | ty::Bool
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
            | ty::Dynamic(_, _)
            => self.pretty_print_type(ty),

            // Placeholders (all printed as `_` to uniformize them).
            | ty::Param(_)
            | ty::Bound(..)
            | ty::Placeholder(_)
            | ty::Infer(_)
            | ty::Error
            => {
                write!(self, "_")?;
                Ok(self)
            }

            // Types with identity (print the module path).
            | ty::Adt(&ty::AdtDef { did: def_id, .. }, substs)
            | ty::FnDef(def_id, substs)
            | ty::Opaque(def_id, substs)
            | ty::Projection(ty::ProjectionTy { item_def_id: def_id, substs })
            | ty::UnnormalizedProjection(ty::ProjectionTy { item_def_id: def_id, substs })
            | ty::Closure(def_id, ty::ClosureSubsts { substs })
            | ty::Generator(def_id, ty::GeneratorSubsts { substs }, _)
            => self.print_def_path(def_id, substs),
            ty::Foreign(def_id) => self.print_def_path(def_id, &[]),

            ty::GeneratorWitness(_) => {
                bug!("type_name: unexpected `GeneratorWitness`")
            }
        }
    }

    fn print_const(
        self,
        _: &'tcx ty::Const<'tcx>,
    ) -> Result<Self::Const, Self::Error> {
        // don't print constants to the user
        Ok(self)
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
        match disambiguated_data.data {
            DefPathData::Ctor => return Ok(self),
            _ => {}
        }

        self.path.push_str("::");

        self.path.push_str(&disambiguated_data.data.as_interned_str().as_str());
        Ok(self)
    }

    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[Kind<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        self = print_prefix(self)?;
        let args = args.iter().cloned().filter(|arg| {
            match arg.unpack() {
                UnpackedKind::Lifetime(_) => false,
                _ => true,
            }
        });
        if args.clone().next().is_some() {
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(self)
        }
    }
}
impl PrettyPrinter<'tcx> for AbsolutePathPrinter<'tcx> {
    fn region_should_not_be_omitted(
        &self,
        _region: ty::Region<'_>,
    ) -> bool {
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
        Ok(self.path.push_str(s))
    }
}

/// Produces an absolute path representation of the given type. See also the documentation on
/// `std::any::type_name`
pub fn type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> &'tcx ty::Const<'tcx> {
    let alloc = alloc_type_name(tcx, ty);
    tcx.mk_const(ty::Const {
        val: ConstValue::Slice {
            data: alloc,
            start: 0,
            end: alloc.bytes.len(),
        },
        ty: tcx.mk_static_str(),
    })
}

/// Directly returns an `Allocation` containing an absolute path representation of the given type.
pub(super) fn alloc_type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> &'tcx Allocation {
    let path = AbsolutePathPrinter { tcx, path: String::new() }.print_type(ty).unwrap().path;
    let alloc = Allocation::from_byte_aligned_bytes(path.into_bytes());
    tcx.intern_const_alloc(alloc)
}
