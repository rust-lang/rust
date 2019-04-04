use std_mangle_rs::ast;

use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId};
use rustc::hir::map::{DefPathData, DisambiguatedDefPathData};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::print::{Printer, Print};
use rustc::ty::subst::{Kind, UnpackedKind};
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
        version: None,
        path: SymbolPrinter { tcx }
            .print_def_path(instance.def_id(), instance.substs)?,
        instantiating_crate: match instantiating_crate {
            Some(instantiating_crate) => Some(
                SymbolPrinter { tcx }
                    .path_crate(instantiating_crate)?
            ),
            None => None,
        },
    };

    let _ = symbol;
    unimplemented!("missing compressor/mangler for mw symbol mangling");

    /*let mut uncompressed = String::new();
    symbol.mangle(&mut uncompressed);

    let (compressed_symbol, _) = std_mangle_rs::compress::compress_ext(&symbol);
    let mut compressed = String::new();
    compressed_symbol.mangle(&mut compressed);

    Ok((uncompressed, compressed))*/
}

#[derive(Copy, Clone)]
struct SymbolPrinter<'a, 'tcx> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl Printer<'tcx, 'tcx> for SymbolPrinter<'_, 'tcx> {
    type Error = Unsupported;

    type Path = ast::Path;
    type Region = ast::Lifetime;
    type Type = ast::Type;
    type DynExistential = ast::DynBounds;
    type Const = ast::Const;

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
        region: ty::Region<'_>,
    ) -> Result<Self::Region, Self::Error> {
        let i = match *region {
            ty::ReErased => 0,

            // FIXME(eddyb) copy the implementation over to here.
            ty::ReLateBound(_, ty::BrAnon(_)) => {
                return Err(Unsupported);
            }

            _ => bug!("mw: non-erased region `{:?}`", region),
        };
        Ok(ast::Lifetime {
            debruijn_index: ast::Base62Number(i),
        })
    }

    fn print_type(
        self,
        ty: Ty<'tcx>,
    ) -> Result<Self::Type, Self::Error> {
        macro_rules! basic {
            ($name:ident) => (ast::Type::BasicType(ast::BasicType::$name))
        }
        Ok(match ty.sty {
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

            // Placeholders (should be demangled as `_`).
            ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) |
            ty::Infer(_) | ty::Error => basic!(Placeholder),

            ty::Ref(r, ty, mutbl) => {
                let lt = if *r != ty::ReErased {
                    Some(r.print(self)?)
                } else {
                    None
                };
                let ty = Arc::new(ty.print(self)?);
                match mutbl {
                    hir::MutImmutable => ast::Type::Ref(lt, ty),
                    hir::MutMutable => ast::Type::RefMut(lt, ty),
                }
            }

            ty::RawPtr(ty::TypeAndMut { ty, mutbl: hir::MutImmutable }) => {
                ast::Type::RawPtrConst(Arc::new(ty.print(self)?))
            }
            ty::RawPtr(ty::TypeAndMut { ty, mutbl: hir::MutMutable }) => {
                ast::Type::RawPtrMut(Arc::new(ty.print(self)?))
            }

            ty::Array(ty, len) => {
                ast::Type::Array(Arc::new(ty.print(self)?), Arc::new(len.print(self)?))
            }
            ty::Slice(ty) => ast::Type::Slice(Arc::new(ty.print(self)?)),

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
                ast::Type::Named(Arc::new(self.print_def_path(def_id, substs)?))
            }
            ty::Foreign(def_id) => {
                ast::Type::Named(Arc::new(self.print_def_path(def_id, &[])?))
            }

            ty::FnPtr(sig) => {
                let mut param_types = sig.inputs().skip_binder().iter()
                    .map(|ty| ty.print(self))
                    .collect::<Result<Vec<_>, _>>()?;
                if sig.c_variadic() {
                    param_types.push(basic!(Ellipsis));
                }
                let return_type = sig.output().skip_binder().print(self)?;
                ast::Type::Fn(Arc::new(ast::FnSig {
                    binder: ast::Binder {
                        // FIXME(eddyb) needs to be implemented, see `print_region`.
                        count: ast::Base62Number(0),
                    },
                    is_unsafe: sig.unsafety() == hir::Unsafety::Unsafe,
                    abi: match sig.abi() {
                        Abi::Rust => None,
                        Abi::C => Some(ast::Abi::C),
                        abi => Some(ast::Abi::Named(ast::UIdent(abi.name().replace('-', "_")))),
                    },
                    param_types,
                    return_type,
                }))
            }

            ty::Dynamic(predicates, r) => {
                let bounds = Arc::new(self.print_dyn_existential(predicates.skip_binder())?);
                let lt = r.print(self)?;
                ast::Type::DynTrait(bounds, lt)
            }

            ty::GeneratorWitness(_) => {
                bug!("mw: unexpected `GeneratorWitness`")
            }
        })
    }

    fn print_dyn_existential(
        self,
        predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        let mut traits = vec![];
        for predicate in predicates {
            match *predicate {
                ty::ExistentialPredicate::Trait(trait_ref) => {
                    // Use a type that can't appear in defaults of type parameters.
                    let dummy_self = self.tcx.mk_infer(ty::FreshTy(0));
                    let trait_ref = trait_ref.with_self_ty(self.tcx, dummy_self);
                    traits.push(ast::DynTrait {
                        path: self.print_def_path(trait_ref.def_id, trait_ref.substs)?,
                        assoc_type_bindings: vec![],
                    });
                }
                ty::ExistentialPredicate::Projection(projection) => {
                    let name = self.tcx.associated_item(projection.item_def_id).ident;
                    traits.last_mut().unwrap().assoc_type_bindings.push(ast::DynTraitAssocBinding {
                        ident: ast::UIdent(name.to_string()),
                        ty: projection.ty.print(self)?,
                    });
                }
                ty::ExistentialPredicate::AutoTrait(def_id) => {
                    traits.push(ast::DynTrait {
                        path: self.print_def_path(def_id, &[])?,
                        assoc_type_bindings: vec![],
                    });
                }
            }
        }

        Ok(ast::DynBounds {
            binder: ast::Binder {
                // FIXME(eddyb) needs to be implemented, see `print_region`.
                count: ast::Base62Number(0),
            },
            traits,
        })
    }

    fn print_const(
        self,
        ct: &'tcx ty::Const<'tcx>,
    ) -> Result<Self::Const, Self::Error> {
        match ct.ty.sty {
            ty::Uint(_) => {}
            _ => {
                bug!("mw: unsupported constant of type `{}` ({:?})",
                    ct.ty, ct);
            }
        }
        let ty = ct.ty.print(self)?;

        if let Some(bits) = ct.assert_bits(self.tcx, ty::ParamEnv::empty().and(ct.ty)) {
            if bits as u64 as u128 != bits {
                return Err(Unsupported);
            }
            Ok(ast::Const::Value(ty, bits as u64))
        } else {
            // NOTE(eddyb) despite having the path, we need to
            // encode a placeholder, as the path could refer
            // back to e.g. an `impl` using the constant.
            Ok(ast::Const::Placeholder(ty))
        }
    }

    fn path_crate(
        self,
        cnum: CrateNum,
    ) -> Result<Self::Path, Self::Error> {
        let fingerprint = self.tcx.crate_disambiguator(cnum).to_fingerprint();
        Ok(ast::Path::CrateRoot {
            id: ast::Ident {
                dis: ast::Base62Number(fingerprint.to_smaller_hash()),
                u_ident: ast::UIdent(self.tcx.original_crate_name(cnum).to_string()),
            },
        })
    }
    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        assert!(trait_ref.is_some());
        let trait_ref = trait_ref.unwrap();

        // This is a default method in the trait declaration.
        Ok(ast::Path::TraitDef {
            self_type: self_ty.print(self)?,
            trait_name: Arc::new(self.print_def_path(trait_ref.def_id, trait_ref.substs)?),
        })
    }

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        let impl_path = ast::ImplPath {
            dis: Some(ast::Base62Number(disambiguated_data.disambiguator as u64)),
            path: Arc::new(print_prefix(self)?),
        };
        let self_type = self_ty.print(self)?;
        match trait_ref {
            Some(trait_ref) => Ok(ast::Path::TraitImpl {
                impl_path,
                self_type,
                trait_name: Arc::new(self.print_def_path(trait_ref.def_id, trait_ref.substs)?),
            }),
            None => Ok(ast::Path::InherentImpl {
                impl_path,
                self_type,
            }),
        }
    }
    fn path_append(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error> {
        let inner = Arc::new(print_prefix(self)?);

        let name = disambiguated_data.data.get_opt_name().map(|s| s.as_str());
        let name = name.as_ref().map_or("", |s| &s[..]);
        let ns = match disambiguated_data.data {
            DefPathData::ClosureExpr => ast::Namespace(b'C'),

            // Lowercase a-z are unspecified disambiguation categories.
            _ => {
                let discriminant = unsafe {
                    ::std::intrinsics::discriminant_value(&disambiguated_data.data)
                };
                assert!(discriminant < 26);

                // Mix in the name to avoid making it too predictable.
                let mut d = (discriminant ^ 0x55) % 26;
                for (i, b) in name.bytes().enumerate() {
                    d = (d + i as u64 + b as u64) % 26;
                }

                ast::Namespace(b'a' + d as u8)
            }
        };

        Ok(ast::Path::Nested {
            ns,
            inner,
            ident: ast::Ident {
                dis: ast::Base62Number(disambiguated_data.disambiguator as u64),
                u_ident: ast::UIdent(name.to_string()),
            }
        })
    }
    fn path_generic_args(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[Kind<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        let prefix = print_prefix(self)?;

        // Don't print any regions if they're all erased.
        let print_regions = args.iter().any(|arg| {
            match arg.unpack() {
                UnpackedKind::Lifetime(r) => *r != ty::ReErased,
                _ => false,
            }
        });
        let args = args.iter().cloned().filter(|arg| {
            match arg.unpack() {
                UnpackedKind::Lifetime(_) => print_regions,
                _ => true,
            }
        });

        if args.clone().next().is_none() {
            return Ok(prefix);
        }

        let args = args.map(|arg| {
            Ok(match arg.unpack() {
                UnpackedKind::Lifetime(lt) => {
                    ast::GenericArg::Lifetime(lt.print(self)?)
                }
                UnpackedKind::Type(ty) => {
                    ast::GenericArg::Type(ty.print(self)?)
                }
                UnpackedKind::Const(ct) => {
                    ast::GenericArg::Const(ct.print(self)?)
                }
            })
        }).collect::<Result<Vec<_>, _>>()?;

        Ok(ast::Path::Generic {
            inner: Arc::new(prefix),
            args,
        })
    }
}
