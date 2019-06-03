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

struct AbsolutePathPrinter<'a, 'tcx> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    path: String,
}

impl<'tcx> Printer<'tcx, 'tcx> for AbsolutePathPrinter<'_, 'tcx> {
    type Error = std::fmt::Error;

    type Path = Self;
    type Region = Self;
    type Type = Self;
    type DynExistential = Self;
    type Const = Self;

    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    fn print_region(self, _region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
        Ok(self)
    }

    fn print_type(self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
        match ty.sty {
            // types without identity
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
            | ty::Adt(..)
            | ty::Foreign(_)
            // should be unreachable, but there's no hurt in printing it (and better than ICEing)
            | ty::Error
            => self.pretty_print_type(ty),
            | ty::Infer(_)
            | ty::Bound(_, _)
            | ty::Param(_)
            | ty::Placeholder(_)
            | ty::Projection(_)
            | ty::UnnormalizedProjection(_)
            | ty::GeneratorWitness(_)
            => bug!(
                "{:#?} in `type_name` should not happen because we are always monomorphized",
                ty,
            ),
            // types with identity (print the module path instead)
            | ty::FnDef(did, substs)
            | ty::Opaque(did, substs)
            => self.print_def_path(did, substs),
            ty::Closure(did, substs) => self.print_def_path(did, substs.substs),
            ty::Generator(did, substs, _) => self.print_def_path(did, substs.substs),
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
impl PrettyPrinter<'tcx, 'tcx> for AbsolutePathPrinter<'_, 'tcx> {
    fn region_should_not_be_omitted(
        &self,
        _region: ty::Region<'_>,
    ) -> bool {
        false
    }
    fn comma_sep<T>(
        mut self,
        mut elems: impl Iterator<Item = T>,
    ) -> Result<Self, Self::Error>
        where T: Print<'tcx, 'tcx, Self, Output = Self, Error = Self::Error>
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

impl Write for AbsolutePathPrinter<'_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        Ok(self.path.push_str(s))
    }
}

/// Produces an absolute path representation of the given type. See also the documentation on
/// `std::any::type_name`
pub fn type_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> &'tcx ty::Const<'tcx> {
    let path = AbsolutePathPrinter { tcx, path: String::new() }.print_type(ty).unwrap().path;
    let len = path.len();
    let alloc = Allocation::from_byte_aligned_bytes(path.into_bytes());
    let alloc = tcx.intern_const_alloc(alloc);
    tcx.mk_const(ty::Const {
        val: ConstValue::Slice {
            data: alloc,
            start: 0,
            end: len,
        },
        ty: tcx.mk_static_str(),
    })
}
