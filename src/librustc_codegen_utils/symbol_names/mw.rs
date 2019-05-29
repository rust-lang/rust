use std_mangle_rs::{ast, compress::compress_ext};

use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId};
use rustc::hir::map::{DefPathData, DisambiguatedDefPathData};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::print::{Printer, Print};
use rustc::ty::subst::{Kind, UnpackedKind};
use rustc_data_structures::base_n;
use rustc_mir::monomorphize::Instance;
use rustc_target::spec::abi::Abi;
use syntax::ast::{IntTy, UintTy, FloatTy};

use std::sync::Arc;

pub(super) struct Unsupported;

pub(super) fn mangle(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    instantiating_crate: Option<CrateNum>,
) -> Result<(String, String), Unsupported> {
    if instance.is_vtable_shim() {
        return Err(Unsupported);
    }

    let symbol = ast::Symbol {
        name: SymbolPrinter { tcx }
            .print_def_path(instance.def_id(), instance.substs)?,
        instantiating_crate: instantiating_crate.map(|instantiating_crate| {
            let fingerprint = tcx.crate_disambiguator(instantiating_crate).to_fingerprint();
            Arc::new(ast::PathPrefix::CrateId {
                name: tcx.original_crate_name(instantiating_crate).to_string(),
                dis: base_n::encode(fingerprint.to_smaller_hash() as u128, 62),
            })
        }),
    };

    let mut uncompressed = String::new();
    symbol.mangle(&mut uncompressed);

    let (compressed_symbol, _) = compress_ext(&symbol);
    let mut compressed = String::new();
    compressed_symbol.mangle(&mut compressed);

    Ok((uncompressed, compressed))
}

#[derive(Copy, Clone)]
struct SymbolPrinter<'a, 'tcx> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl Printer<'tcx, 'tcx> for SymbolPrinter<'_, 'tcx> {
    type Error = Unsupported;

    type Path = Arc<ast::AbsolutePath>;
    type Region = !;
    type Type = Arc<ast::Type>;
    type DynExistential = !;
    type Const = !;

    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    fn print_impl_path(
        self,
        impl_def_id: DefId,
        _substs: &[Kind<'tcx>],
        self_ty: Ty<'tcx>,
        impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        let key = self.tcx.def_key(impl_def_id);
        let parent_def_id = DefId { index: key.parent.unwrap(), ..impl_def_id };

        self.path_append_impl(
            |cx| cx.print_def_path(parent_def_id, &[]),
            &key.disambiguated_data,
            self_ty,
            impl_trait_ref,
        )
    }

    fn print_region(
        self,
        _region: ty::Region<'_>,
    ) -> Result<Self::Region, Self::Error> {
        bug!("mw::print_region: should never be called")
    }

    fn print_type(
        self,
        ty: Ty<'tcx>,
    ) -> Result<Self::Type, Self::Error> {
        macro_rules! basic {
            ($name:ident) => (ast::Type::BasicType(ast::BasicType::$name))
        }
        Ok(Arc::new(match ty.sty {
            ty::Bool => basic!(Bool),
            ty::Char => basic!(Char),
            ty::Str => basic!(Str),
            ty::Tuple(_) if ty.is_unit() => basic!(Unit),
            ty::Int(IntTy::I8) => basic!(I8),
            ty::Int(IntTy::I16) => basic!(I16),
            ty::Int(IntTy::I32) => basic!(I32),
            ty::Int(IntTy::I64) => basic!(I64),
            ty::Int(IntTy::I128) => basic!(I128),
            ty::Int(IntTy::Isize) => basic!(Isize),
            ty::Uint(UintTy::U8) => basic!(U8),
            ty::Uint(UintTy::U16) => basic!(U16),
            ty::Uint(UintTy::U32) => basic!(U32),
            ty::Uint(UintTy::U64) => basic!(U64),
            ty::Uint(UintTy::U128) => basic!(U128),
            ty::Uint(UintTy::Usize) => basic!(Usize),
            ty::Float(FloatTy::F32) => basic!(F32),
            ty::Float(FloatTy::F64) => basic!(F64),
            ty::Never => basic!(Never),

            ty::Ref(_, ty, hir::MutImmutable) => ast::Type::Ref(ty.print(self)?),
            ty::Ref(_, ty, hir::MutMutable) => ast::Type::RefMut(ty.print(self)?),

            ty::RawPtr(ty::TypeAndMut { ty, mutbl: hir::MutImmutable }) => {
                ast::Type::RawPtrConst(ty.print(self)?)
            }
            ty::RawPtr(ty::TypeAndMut { ty, mutbl: hir::MutMutable }) => {
                ast::Type::RawPtrMut(ty.print(self)?)
            }

            ty::Array(ty, len) => {
                let len = len.assert_usize(self.tcx()).ok_or(Unsupported)?;
                ast::Type::Array(Some(len), ty.print(self)?)
            }
            ty::Slice(ty) => ast::Type::Array(None, ty.print(self)?),

            ty::Tuple(tys) => {
                let tys = tys.iter()
                    .map(|k| k.expect_ty().print(self))
                    .collect::<Result<Vec<_>, _>>()?;
                ast::Type::Tuple(tys)
            }

            // Mangle all nominal types as paths.
            ty::Adt(&ty::AdtDef { did: def_id, .. }, substs) |
            ty::FnDef(def_id, substs) |
            ty::Opaque(def_id, substs) |
            ty::Projection(ty::ProjectionTy { item_def_id: def_id, substs }) |
            ty::UnnormalizedProjection(ty::ProjectionTy { item_def_id: def_id, substs }) |
            ty::Closure(def_id, ty::ClosureSubsts { substs }) |
            ty::Generator(def_id, ty::GeneratorSubsts { substs }, _) => {
                ast::Type::Named(self.print_def_path(def_id, substs)?)
            }
            ty::Foreign(def_id) => {
                ast::Type::Named(self.print_def_path(def_id, &[])?)
            }

            ty::Param(p) => ast::Type::GenericParam(ast::Ident {
                ident: p.name.to_string(),
                tag: ast::IdentTag::TypeNs,
                dis: ast::NumericDisambiguator(0),
            }),

            ty::FnPtr(sig) => {
                let mut params = sig.inputs().skip_binder().iter()
                    .map(|ty| ty.print(self))
                    .collect::<Result<Vec<_>, _>>()?;
                if sig.c_variadic() {
                    params.push(Arc::new(basic!(Ellipsis)));
                }
                let output = *sig.output().skip_binder();
                let return_type = if output.is_unit() {
                    None
                } else {
                    Some(output.print(self)?)
                };
                ast::Type::Fn {
                    is_unsafe: sig.unsafety() == hir::Unsafety::Unsafe,
                    abi: match sig.abi() {
                        Abi::Rust => ast::Abi::Rust,
                        Abi::C => ast::Abi::C,
                        _ => return Err(Unsupported),
                    },
                    params,
                    return_type,
                }
            }

            _ => return Err(Unsupported),
        }))
    }

    fn print_dyn_existential(
        self,
        _predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        Err(Unsupported)
    }

    fn print_const(
        self,
        _ct: &'tcx ty::Const<'tcx>,
    ) -> Result<Self::Const, Self::Error> {
        Err(Unsupported)
    }

    fn path_crate(
        self,
        cnum: CrateNum,
    ) -> Result<Self::Path, Self::Error> {
        let fingerprint = self.tcx.crate_disambiguator(cnum).to_fingerprint();
        let path = ast::PathPrefix::CrateId {
            name: self.tcx.original_crate_name(cnum).to_string(),
            dis: base_n::encode(fingerprint.to_smaller_hash() as u128, 62),
        };
        Ok(Arc::new(ast::AbsolutePath::Path {
            name: Arc::new(path),
            args: ast::GenericArgumentList::new_empty(),
        }))
    }
    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        assert!(trait_ref.is_some());
        let trait_ref = trait_ref.unwrap();

        // This is a default method in the trait declaration.
        let path = ast::PathPrefix::TraitImpl {
            self_type: self_ty.print(self)?,
            impled_trait: Some(self.print_def_path(trait_ref.def_id, trait_ref.substs)?),
            dis: ast::NumericDisambiguator(0),
        };
        Ok(Arc::new(ast::AbsolutePath::Path {
            name: Arc::new(path),
            args: ast::GenericArgumentList::new_empty(),
        }))
    }

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        let path = ast::PathPrefix::TraitImpl {
            // HACK(eddyb) include the `impl` prefix into the path, by nesting
            // another `TraitImpl` node into the Self type of the `impl`, e.g.:
            // `foo::<impl Tr for X>::..` becomes `<<X as foo> as Tr>::...`.
            self_type: Arc::new(ast::Type::Named(Arc::new(ast::AbsolutePath::Path {
                name: Arc::new(ast::PathPrefix::TraitImpl {
                    self_type: self_ty.print(self)?,
                    impled_trait: Some(print_prefix(self)?),
                    dis: ast::NumericDisambiguator(disambiguated_data.disambiguator as u64),
                }),
                args: ast::GenericArgumentList::new_empty(),
            }))),

            impled_trait: match trait_ref {
                Some(trait_ref) => Some(
                    self.print_def_path(trait_ref.def_id, trait_ref.substs)?
                ),
                None => None,
            },
            dis: ast::NumericDisambiguator(0),
        };
        Ok(Arc::new(ast::AbsolutePath::Path {
            name: Arc::new(path),
            args: ast::GenericArgumentList::new_empty(),
        }))
    }
    fn path_append(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error> {
        let mut path = print_prefix(self)?;

        let (prefix, ast_args) = match Arc::make_mut(&mut path) {
            ast::AbsolutePath::Path { name, args } => (name, args),
            _ => unreachable!(),
        };

        let mut ident = match disambiguated_data.data {
            DefPathData::ClosureExpr => String::new(),
            _ => disambiguated_data.data.get_opt_name().ok_or(Unsupported)?.to_string(),
        };

        let tag = match disambiguated_data.data {
            DefPathData::ClosureExpr => ast::IdentTag::Closure,

            /*DefPathData::ValueNs(..) |
            DefPathData::Ctor |
            DefPathData::Field(..) => ast::IdentTag::ValueNs,*/

            // HACK(eddyb) rather than using `ValueNs` (see above), this
            // encodes the disambiguated category into the identifier, so it's
            // lossless (see the RFC for why we can't just do type vs value).
            _ => {
                let tag = {
                    let discriminant = unsafe {
                        ::std::intrinsics::discriminant_value(&disambiguated_data.data)
                    };
                    assert!(discriminant < 26);

                    // Mix in the name to avoid making it too predictable.
                    let mut d = (discriminant ^ 0x55) % 26;
                    for (i, b) in ident.bytes().enumerate() {
                        d = (d + i as u64 + b as u64) % 26;
                    }

                    (b'A' + d as u8) as char
                };
                ident.push(tag);

                ast::IdentTag::TypeNs
            }
        };

        let dis = ast::NumericDisambiguator(disambiguated_data.disambiguator as u64);

        let prefix = if !ast_args.is_empty() {
            Arc::new(ast::PathPrefix::AbsolutePath { path })
        } else {
            prefix.clone()
        };

        Ok(Arc::new(ast::AbsolutePath::Path {
            name: Arc::new(ast::PathPrefix::Node {
                prefix: prefix.clone(),
                ident: ast::Ident { ident, tag, dis },
            }),
            args: ast::GenericArgumentList::new_empty(),
        }))
    }
    fn path_generic_args(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[Kind<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        let mut path = print_prefix(self)?;

        if args.is_empty() {
            return Ok(path);
        }

        let ast_args = match Arc::make_mut(&mut path) {
            ast::AbsolutePath::Path { args, .. } => args,
            _ => unreachable!(),
        };

        if !ast_args.is_empty() {
            bug!("mw::path_generic_args({:?}): prefix already has generic args: {:#?}",
                args, path);
        }

        for &arg in args {
            match arg.unpack() {
                UnpackedKind::Lifetime(_) => {}
                UnpackedKind::Type(ty) => {
                    ast_args.0.push(ty.print(self)?);
                }
                UnpackedKind::Const(_) => return Err(Unsupported),
            }
        }

        Ok(path)
    }
}
