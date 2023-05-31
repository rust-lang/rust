use rustc_data_structures::base_n;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::intern::Interned;
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::print::{Print, Printer};
use rustc_middle::ty::{
    self, EarlyBinder, FloatTy, Instance, IntTy, Ty, TyCtxt, TypeVisitable, TypeVisitableExt,
    UintTy,
};
use rustc_middle::ty::{GenericArg, GenericArgKind};
use rustc_span::symbol::kw;
use rustc_target::abi::Integer;
use rustc_target::spec::abi::Abi;

use std::fmt::Write;
use std::iter;
use std::ops::Range;

pub(super) fn mangle<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    instantiating_crate: Option<CrateNum>,
) -> String {
    let def_id = instance.def_id();
    // FIXME(eddyb) this should ideally not be needed.
    let substs = tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), instance.substs);

    let prefix = "_R";
    let mut cx = &mut SymbolMangler {
        tcx,
        start_offset: prefix.len(),
        paths: FxHashMap::default(),
        types: FxHashMap::default(),
        consts: FxHashMap::default(),
        binders: vec![],
        out: String::from(prefix),
    };

    // Append `::{shim:...#0}` to shims that can coexist with a non-shim instance.
    let shim_kind = match instance.def {
        ty::InstanceDef::ThreadLocalShim(_) => Some("tls"),
        ty::InstanceDef::VTableShim(_) => Some("vtable"),
        ty::InstanceDef::ReifyShim(_) => Some("reify"),

        _ => None,
    };

    cx = if let Some(shim_kind) = shim_kind {
        cx.path_append_ns(|cx| cx.print_def_path(def_id, substs), 'S', 0, shim_kind).unwrap()
    } else {
        cx.print_def_path(def_id, substs).unwrap()
    };
    if let Some(instantiating_crate) = instantiating_crate {
        cx = cx.print_def_path(instantiating_crate.as_def_id(), &[]).unwrap();
    }
    std::mem::take(&mut cx.out)
}

pub(super) fn mangle_typeid_for_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyExistentialTraitRef<'tcx>,
) -> String {
    // FIXME(flip1995): See comment in `mangle_typeid_for_fnabi`.
    let mut cx = &mut SymbolMangler {
        tcx,
        start_offset: 0,
        paths: FxHashMap::default(),
        types: FxHashMap::default(),
        consts: FxHashMap::default(),
        binders: vec![],
        out: String::new(),
    };
    cx = cx.print_def_path(trait_ref.def_id(), &[]).unwrap();
    std::mem::take(&mut cx.out)
}

struct BinderLevel {
    /// The range of distances from the root of what's
    /// being printed, to the lifetimes in a binder.
    /// Specifically, a `BrAnon` lifetime has depth
    /// `lifetime_depths.start + index`, going away from the
    /// the root and towards its use site, as the var index increases.
    /// This is used to flatten rustc's pairing of `BrAnon`
    /// (intra-binder disambiguation) with a `DebruijnIndex`
    /// (binder addressing), to "true" de Bruijn indices,
    /// by subtracting the depth of a certain lifetime, from
    /// the innermost depth at its use site.
    lifetime_depths: Range<u32>,
}

struct SymbolMangler<'tcx> {
    tcx: TyCtxt<'tcx>,
    binders: Vec<BinderLevel>,
    out: String,

    /// The length of the prefix in `out` (e.g. 2 for `_R`).
    start_offset: usize,
    /// The values are start positions in `out`, in bytes.
    paths: FxHashMap<(DefId, &'tcx [GenericArg<'tcx>]), usize>,
    types: FxHashMap<Ty<'tcx>, usize>,
    consts: FxHashMap<ty::Const<'tcx>, usize>,
}

impl<'tcx> SymbolMangler<'tcx> {
    fn push(&mut self, s: &str) {
        self.out.push_str(s);
    }

    /// Push a `_`-terminated base 62 integer, using the format
    /// specified in the RFC as `<base-62-number>`, that is:
    /// * `x = 0` is encoded as just the `"_"` terminator
    /// * `x > 0` is encoded as `x - 1` in base 62, followed by `"_"`,
    ///   e.g. `1` becomes `"0_"`, `62` becomes `"Z_"`, etc.
    fn push_integer_62(&mut self, x: u64) {
        if let Some(x) = x.checked_sub(1) {
            base_n::push_str(x as u128, 62, &mut self.out);
        }
        self.push("_");
    }

    /// Push a `tag`-prefixed base 62 integer, when larger than `0`, that is:
    /// * `x = 0` is encoded as `""` (nothing)
    /// * `x > 0` is encoded as the `tag` followed by `push_integer_62(x - 1)`
    ///   e.g. `1` becomes `tag + "_"`, `2` becomes `tag + "0_"`, etc.
    fn push_opt_integer_62(&mut self, tag: &str, x: u64) {
        if let Some(x) = x.checked_sub(1) {
            self.push(tag);
            self.push_integer_62(x);
        }
    }

    fn push_disambiguator(&mut self, dis: u64) {
        self.push_opt_integer_62("s", dis);
    }

    fn push_ident(&mut self, ident: &str) {
        let mut use_punycode = false;
        for b in ident.bytes() {
            match b {
                b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => {}
                0x80..=0xff => use_punycode = true,
                _ => bug!("symbol_names: bad byte {} in ident {:?}", b, ident),
            }
        }

        let punycode_string;
        let ident = if use_punycode {
            self.push("u");

            // FIXME(eddyb) we should probably roll our own punycode implementation.
            let mut punycode_bytes = match punycode::encode(ident) {
                Ok(s) => s.into_bytes(),
                Err(()) => bug!("symbol_names: punycode encoding failed for ident {:?}", ident),
            };

            // Replace `-` with `_`.
            if let Some(c) = punycode_bytes.iter_mut().rfind(|&&mut c| c == b'-') {
                *c = b'_';
            }

            // FIXME(eddyb) avoid rechecking UTF-8 validity.
            punycode_string = String::from_utf8(punycode_bytes).unwrap();
            &punycode_string
        } else {
            ident
        };

        let _ = write!(self.out, "{}", ident.len());

        // Write a separating `_` if necessary (leading digit or `_`).
        if let Some('_' | '0'..='9') = ident.chars().next() {
            self.push("_");
        }

        self.push(ident);
    }

    fn path_append_ns<'a>(
        mut self: &'a mut Self,
        print_prefix: impl FnOnce(&'a mut Self) -> Result<&'a mut Self, !>,
        ns: char,
        disambiguator: u64,
        name: &str,
    ) -> Result<&'a mut Self, !> {
        self.push("N");
        self.out.push(ns);
        self = print_prefix(self)?;
        self.push_disambiguator(disambiguator as u64);
        self.push_ident(name);
        Ok(self)
    }

    fn print_backref(&mut self, i: usize) -> Result<&mut Self, !> {
        self.push("B");
        self.push_integer_62((i - self.start_offset) as u64);
        Ok(self)
    }

    fn in_binder<'a, T>(
        mut self: &'a mut Self,
        value: &ty::Binder<'tcx, T>,
        print_value: impl FnOnce(&'a mut Self, &T) -> Result<&'a mut Self, !>,
    ) -> Result<&'a mut Self, !>
    where
        T: TypeVisitable<TyCtxt<'tcx>>,
    {
        let mut lifetime_depths =
            self.binders.last().map(|b| b.lifetime_depths.end).map_or(0..0, |i| i..i);

        // FIXME(non-lifetime-binders): What to do here?
        let lifetimes = value
            .bound_vars()
            .iter()
            .filter(|var| matches!(var, ty::BoundVariableKind::Region(..)))
            .count() as u32;

        self.push_opt_integer_62("G", lifetimes as u64);
        lifetime_depths.end += lifetimes;

        self.binders.push(BinderLevel { lifetime_depths });
        self = print_value(self, value.as_ref().skip_binder())?;
        self.binders.pop();

        Ok(self)
    }
}

impl<'tcx> Printer<'tcx> for &mut SymbolMangler<'tcx> {
    type Error = !;

    type Path = Self;
    type Region = Self;
    type Type = Self;
    type DynExistential = Self;
    type Const = Self;

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_def_path(
        mut self,
        def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        if let Some(&i) = self.paths.get(&(def_id, substs)) {
            return self.print_backref(i);
        }
        let start = self.out.len();

        self = self.default_print_def_path(def_id, substs)?;

        // Only cache paths that do not refer to an enclosing
        // binder (which would change depending on context).
        if !substs.iter().any(|k| k.has_escaping_bound_vars()) {
            self.paths.insert((def_id, substs), start);
        }
        Ok(self)
    }

    fn print_impl_path(
        mut self,
        impl_def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
        mut self_ty: Ty<'tcx>,
        mut impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        let key = self.tcx.def_key(impl_def_id);
        let parent_def_id = DefId { index: key.parent.unwrap(), ..impl_def_id };

        let mut param_env = self.tcx.param_env_reveal_all_normalized(impl_def_id);
        if !substs.is_empty() {
            param_env = EarlyBinder::bind(param_env).subst(self.tcx, substs);
        }

        match &mut impl_trait_ref {
            Some(impl_trait_ref) => {
                assert_eq!(impl_trait_ref.self_ty(), self_ty);
                *impl_trait_ref = self.tcx.normalize_erasing_regions(param_env, *impl_trait_ref);
                self_ty = impl_trait_ref.self_ty();
            }
            None => {
                self_ty = self.tcx.normalize_erasing_regions(param_env, self_ty);
            }
        }

        self.push(match impl_trait_ref {
            Some(_) => "X",
            None => "M",
        });

        // Encode impl generic params if the substitutions contain parameters (implying
        // polymorphization is enabled) and this isn't an inherent impl.
        if impl_trait_ref.is_some() && substs.iter().any(|a| a.has_non_region_param()) {
            self = self.path_generic_args(
                |this| {
                    this.path_append_ns(
                        |cx| cx.print_def_path(parent_def_id, &[]),
                        'I',
                        key.disambiguated_data.disambiguator as u64,
                        "",
                    )
                },
                substs,
            )?;
        } else {
            self.push_disambiguator(key.disambiguated_data.disambiguator as u64);
            self = self.print_def_path(parent_def_id, &[])?;
        }

        self = self_ty.print(self)?;

        if let Some(trait_ref) = impl_trait_ref {
            self = self.print_def_path(trait_ref.def_id, trait_ref.substs)?;
        }

        Ok(self)
    }

    fn print_region(self, region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
        let i = match *region {
            // Erased lifetimes use the index 0, for a
            // shorter mangling of `L_`.
            ty::ReErased => 0,

            // Late-bound lifetimes use indices starting at 1,
            // see `BinderLevel` for more details.
            ty::ReLateBound(debruijn, ty::BoundRegion { var, kind: ty::BrAnon(_) }) => {
                let binder = &self.binders[self.binders.len() - 1 - debruijn.index()];
                let depth = binder.lifetime_depths.start + var.as_u32();

                1 + (self.binders.last().unwrap().lifetime_depths.end - 1 - depth)
            }

            _ => bug!("symbol_names: non-erased region `{:?}`", region),
        };
        self.push("L");
        self.push_integer_62(i as u64);
        Ok(self)
    }

    fn print_type(mut self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
        // Basic types, never cached (single-character).
        let basic_type = match ty.kind() {
            ty::Bool => "b",
            ty::Char => "c",
            ty::Str => "e",
            ty::Tuple(_) if ty.is_unit() => "u",
            ty::Int(IntTy::I8) => "a",
            ty::Int(IntTy::I16) => "s",
            ty::Int(IntTy::I32) => "l",
            ty::Int(IntTy::I64) => "x",
            ty::Int(IntTy::I128) => "n",
            ty::Int(IntTy::Isize) => "i",
            ty::Uint(UintTy::U8) => "h",
            ty::Uint(UintTy::U16) => "t",
            ty::Uint(UintTy::U32) => "m",
            ty::Uint(UintTy::U64) => "y",
            ty::Uint(UintTy::U128) => "o",
            ty::Uint(UintTy::Usize) => "j",
            ty::Float(FloatTy::F32) => "f",
            ty::Float(FloatTy::F64) => "d",
            ty::Never => "z",

            // Placeholders (should be demangled as `_`).
            ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => "p",

            _ => "",
        };
        if !basic_type.is_empty() {
            self.push(basic_type);
            return Ok(self);
        }

        if let Some(&i) = self.types.get(&ty) {
            return self.print_backref(i);
        }
        let start = self.out.len();

        match *ty.kind() {
            // Basic types, handled above.
            ty::Bool | ty::Char | ty::Str | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never => {
                unreachable!()
            }
            ty::Tuple(_) if ty.is_unit() => unreachable!(),

            // Placeholders, also handled as part of basic types.
            ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => {
                unreachable!()
            }

            ty::Ref(r, ty, mutbl) => {
                self.push(match mutbl {
                    hir::Mutability::Not => "R",
                    hir::Mutability::Mut => "Q",
                });
                if !r.is_erased() {
                    self = r.print(self)?;
                }
                self = ty.print(self)?;
            }

            ty::RawPtr(mt) => {
                self.push(match mt.mutbl {
                    hir::Mutability::Not => "P",
                    hir::Mutability::Mut => "O",
                });
                self = mt.ty.print(self)?;
            }

            ty::Array(ty, len) => {
                self.push("A");
                self = ty.print(self)?;
                self = self.print_const(len)?;
            }
            ty::Slice(ty) => {
                self.push("S");
                self = ty.print(self)?;
            }

            ty::Tuple(tys) => {
                self.push("T");
                for ty in tys.iter() {
                    self = ty.print(self)?;
                }
                self.push("E");
            }

            // Mangle all nominal types as paths.
            ty::Adt(ty::AdtDef(Interned(&ty::AdtDefData { did: def_id, .. }, _)), substs)
            | ty::FnDef(def_id, substs)
            | ty::Alias(ty::Projection | ty::Opaque, ty::AliasTy { def_id, substs, .. })
            | ty::Closure(def_id, substs)
            | ty::Generator(def_id, substs, _) => {
                self = self.print_def_path(def_id, substs)?;
            }
            ty::Foreign(def_id) => {
                self = self.print_def_path(def_id, &[])?;
            }

            ty::FnPtr(sig) => {
                self.push("F");
                self = self.in_binder(&sig, |mut cx, sig| {
                    if sig.unsafety == hir::Unsafety::Unsafe {
                        cx.push("U");
                    }
                    match sig.abi {
                        Abi::Rust => {}
                        Abi::C { unwind: false } => cx.push("KC"),
                        abi => {
                            cx.push("K");
                            let name = abi.name();
                            if name.contains('-') {
                                cx.push_ident(&name.replace('-', "_"));
                            } else {
                                cx.push_ident(name);
                            }
                        }
                    }
                    for &ty in sig.inputs() {
                        cx = ty.print(cx)?;
                    }
                    if sig.c_variadic {
                        cx.push("v");
                    }
                    cx.push("E");
                    sig.output().print(cx)
                })?;
            }

            ty::Dynamic(predicates, r, kind) => {
                self.push(match kind {
                    ty::Dyn => "D",
                    // FIXME(dyn-star): need to update v0 mangling docs
                    ty::DynStar => "D*",
                });
                self = self.print_dyn_existential(predicates)?;
                self = r.print(self)?;
            }

            ty::Alias(ty::Inherent, _) => bug!("symbol_names: unexpected inherent projection"),
            ty::GeneratorWitness(_) => bug!("symbol_names: unexpected `GeneratorWitness`"),
            ty::GeneratorWitnessMIR(..) => bug!("symbol_names: unexpected `GeneratorWitnessMIR`"),
        }

        // Only cache types that do not refer to an enclosing
        // binder (which would change depending on context).
        if !ty.has_escaping_bound_vars() {
            self.types.insert(ty, start);
        }
        Ok(self)
    }

    fn print_dyn_existential(
        mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error> {
        // Okay, so this is a bit tricky. Imagine we have a trait object like
        // `dyn for<'a> Foo<'a, Bar = &'a ()>`. When we mangle this, the
        // output looks really close to the syntax, where the `Bar = &'a ()` bit
        // is under the same binders (`['a]`) as the `Foo<'a>` bit. However, we
        // actually desugar these into two separate `ExistentialPredicate`s. We
        // can't enter/exit the "binder scope" twice though, because then we
        // would mangle the binders twice. (Also, side note, we merging these
        // two is kind of difficult, because of potential HRTBs in the Projection
        // predicate.)
        //
        // Also worth mentioning: imagine that we instead had
        // `dyn for<'a> Foo<'a, Bar = &'a ()> + Send`. In this case, `Send` is
        // under the same binders as `Foo`. Currently, this doesn't matter,
        // because only *auto traits* are allowed other than the principal trait
        // and all auto traits don't have any generics. Two things could
        // make this not an "okay" mangling:
        // 1) Instead of mangling only *used*
        // bound vars, we want to mangle *all* bound vars (`for<'b> Send` is a
        // valid trait predicate);
        // 2) We allow multiple "principal" traits in the future, or at least
        // allow in any form another trait predicate that can take generics.
        //
        // Here we assume that predicates have the following structure:
        // [<Trait> [{<Projection>}]] [{<Auto>}]
        // Since any predicates after the first one shouldn't change the binders,
        // just put them all in the binders of the first.
        self = self.in_binder(&predicates[0], |mut cx, _| {
            for predicate in predicates.iter() {
                // It would be nice to be able to validate bound vars here, but
                // projections can actually include bound vars from super traits
                // because of HRTBs (only in the `Self` type). Also, auto traits
                // could have different bound vars *anyways*.
                match predicate.as_ref().skip_binder() {
                    ty::ExistentialPredicate::Trait(trait_ref) => {
                        // Use a type that can't appear in defaults of type parameters.
                        let dummy_self = cx.tcx.mk_fresh_ty(0);
                        let trait_ref = trait_ref.with_self_ty(cx.tcx, dummy_self);
                        cx = cx.print_def_path(trait_ref.def_id, trait_ref.substs)?;
                    }
                    ty::ExistentialPredicate::Projection(projection) => {
                        let name = cx.tcx.associated_item(projection.def_id).name;
                        cx.push("p");
                        cx.push_ident(name.as_str());
                        cx = match projection.term.unpack() {
                            ty::TermKind::Ty(ty) => ty.print(cx),
                            ty::TermKind::Const(c) => c.print(cx),
                        }?;
                    }
                    ty::ExistentialPredicate::AutoTrait(def_id) => {
                        cx = cx.print_def_path(*def_id, &[])?;
                    }
                }
            }
            Ok(cx)
        })?;

        self.push("E");
        Ok(self)
    }

    fn print_const(mut self, ct: ty::Const<'tcx>) -> Result<Self::Const, Self::Error> {
        // We only mangle a typed value if the const can be evaluated.
        let ct = ct.eval(self.tcx, ty::ParamEnv::reveal_all());
        match ct.kind() {
            ty::ConstKind::Value(_) => {}

            // Placeholders (should be demangled as `_`).
            // NOTE(eddyb) despite `Unevaluated` having a `DefId` (and therefore
            // a path), even for it we still need to encode a placeholder, as
            // the path could refer back to e.g. an `impl` using the constant.
            ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Expr(_)
            | ty::ConstKind::Param(_)
            | ty::ConstKind::Infer(_)
            | ty::ConstKind::Bound(..)
            | ty::ConstKind::Placeholder(_)
            | ty::ConstKind::Error(_) => {
                // Never cached (single-character).
                self.push("p");
                return Ok(self);
            }
        }

        if let Some(&i) = self.consts.get(&ct) {
            return self.print_backref(i);
        }

        let start = self.out.len();
        let ty = ct.ty();

        match ty.kind() {
            ty::Uint(_) | ty::Int(_) | ty::Bool | ty::Char => {
                self = ty.print(self)?;

                let mut bits = ct.eval_bits(self.tcx, ty::ParamEnv::reveal_all(), ty);

                // Negative integer values are mangled using `n` as a "sign prefix".
                if let ty::Int(ity) = ty.kind() {
                    let val =
                        Integer::from_int_ty(&self.tcx, *ity).size().sign_extend(bits) as i128;
                    if val < 0 {
                        self.push("n");
                    }
                    bits = val.unsigned_abs();
                }

                let _ = write!(self.out, "{bits:x}_");
            }

            // FIXME(valtrees): Remove the special case for `str`
            // here and fully support unsized constants.
            ty::Ref(_, inner_ty, mutbl) => {
                self.push(match mutbl {
                    hir::Mutability::Not => "R",
                    hir::Mutability::Mut => "Q",
                });

                match inner_ty.kind() {
                    ty::Str if mutbl.is_not() => {
                        match ct.kind() {
                            ty::ConstKind::Value(valtree) => {
                                let slice =
                                    valtree.try_to_raw_bytes(self.tcx(), ty).unwrap_or_else(|| {
                                        bug!(
                                        "expected to get raw bytes from valtree {:?} for type {:}",
                                        valtree, ty
                                    )
                                    });
                                let s = std::str::from_utf8(slice).expect("non utf8 str from miri");

                                self.push("e");

                                // FIXME(eddyb) use a specialized hex-encoding loop.
                                for byte in s.bytes() {
                                    let _ = write!(self.out, "{byte:02x}");
                                }

                                self.push("_");
                            }

                            _ => {
                                bug!("symbol_names: unsupported `&str` constant: {:?}", ct);
                            }
                        }
                    }
                    _ => {
                        let pointee_ty = ct
                            .ty()
                            .builtin_deref(true)
                            .expect("tried to dereference on non-ptr type")
                            .ty;
                        let dereferenced_const = self.tcx.mk_const(ct.kind(), pointee_ty);
                        self = dereferenced_const.print(self)?;
                    }
                }
            }

            ty::Array(..) | ty::Tuple(..) | ty::Adt(..) | ty::Slice(_) => {
                let contents = self.tcx.destructure_const(ct);
                let fields = contents.fields.iter().copied();

                let print_field_list = |mut this: Self| {
                    for field in fields.clone() {
                        this = field.print(this)?;
                    }
                    this.push("E");
                    Ok(this)
                };

                match *ct.ty().kind() {
                    ty::Array(..) | ty::Slice(_) => {
                        self.push("A");
                        self = print_field_list(self)?;
                    }
                    ty::Tuple(..) => {
                        self.push("T");
                        self = print_field_list(self)?;
                    }
                    ty::Adt(def, substs) => {
                        let variant_idx =
                            contents.variant.expect("destructed const of adt without variant idx");
                        let variant_def = &def.variant(variant_idx);

                        self.push("V");
                        self = self.print_def_path(variant_def.def_id, substs)?;

                        match variant_def.ctor_kind() {
                            Some(CtorKind::Const) => {
                                self.push("U");
                            }
                            Some(CtorKind::Fn) => {
                                self.push("T");
                                self = print_field_list(self)?;
                            }
                            None => {
                                self.push("S");
                                for (field_def, field) in iter::zip(&variant_def.fields, fields) {
                                    // HACK(eddyb) this mimics `path_append`,
                                    // instead of simply using `field_def.ident`,
                                    // just to be able to handle disambiguators.
                                    let disambiguated_field =
                                        self.tcx.def_key(field_def.did).disambiguated_data;
                                    let field_name = disambiguated_field.data.get_opt_name();
                                    self.push_disambiguator(
                                        disambiguated_field.disambiguator as u64,
                                    );
                                    self.push_ident(field_name.unwrap_or(kw::Empty).as_str());

                                    self = field.print(self)?;
                                }
                                self.push("E");
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => {
                bug!("symbol_names: unsupported constant of type `{}` ({:?})", ct.ty(), ct);
            }
        }

        // Only cache consts that do not refer to an enclosing
        // binder (which would change depending on context).
        if !ct.has_escaping_bound_vars() {
            self.consts.insert(ct, start);
        }
        Ok(self)
    }

    fn path_crate(self, cnum: CrateNum) -> Result<Self::Path, Self::Error> {
        self.push("C");
        let stable_crate_id = self.tcx.def_path_hash(cnum.as_def_id()).stable_crate_id();
        self.push_disambiguator(stable_crate_id.as_u64());
        let name = self.tcx.crate_name(cnum);
        self.push_ident(name.as_str());
        Ok(self)
    }

    fn path_qualified(
        mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        assert!(trait_ref.is_some());
        let trait_ref = trait_ref.unwrap();

        self.push("Y");
        self = self_ty.print(self)?;
        self.print_def_path(trait_ref.def_id, trait_ref.substs)
    }

    fn path_append_impl(
        self,
        _: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        _: &DisambiguatedDefPathData,
        _: Ty<'tcx>,
        _: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        // Inlined into `print_impl_path`
        unreachable!()
    }

    fn path_append(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error> {
        let ns = match disambiguated_data.data {
            // Extern block segments can be skipped, names from extern blocks
            // are effectively living in their parent modules.
            DefPathData::ForeignMod => return print_prefix(self),

            // Uppercase categories are more stable than lowercase ones.
            DefPathData::TypeNs(_) => 't',
            DefPathData::ValueNs(_) => 'v',
            DefPathData::ClosureExpr => 'C',
            DefPathData::Ctor => 'c',
            DefPathData::AnonConst => 'k',
            DefPathData::ImplTrait => 'i',

            // These should never show up as `path_append` arguments.
            DefPathData::CrateRoot
            | DefPathData::Use
            | DefPathData::GlobalAsm
            | DefPathData::Impl
            | DefPathData::ImplTraitAssocTy
            | DefPathData::MacroNs(_)
            | DefPathData::LifetimeNs(_) => {
                bug!("symbol_names: unexpected DefPathData: {:?}", disambiguated_data.data)
            }
        };

        let name = disambiguated_data.data.get_opt_name();

        self.path_append_ns(
            print_prefix,
            ns,
            disambiguated_data.disambiguator as u64,
            name.unwrap_or(kw::Empty).as_str(),
        )
    }

    fn path_generic_args(
        mut self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        // Don't print any regions if they're all erased.
        let print_regions = args.iter().any(|arg| match arg.unpack() {
            GenericArgKind::Lifetime(r) => !r.is_erased(),
            _ => false,
        });
        let args = args.iter().cloned().filter(|arg| match arg.unpack() {
            GenericArgKind::Lifetime(_) => print_regions,
            _ => true,
        });

        if args.clone().next().is_none() {
            return print_prefix(self);
        }

        self.push("I");
        self = print_prefix(self)?;
        for arg in args {
            match arg.unpack() {
                GenericArgKind::Lifetime(lt) => {
                    self = lt.print(self)?;
                }
                GenericArgKind::Type(ty) => {
                    self = ty.print(self)?;
                }
                GenericArgKind::Const(c) => {
                    self.push("K");
                    self = c.print(self)?;
                }
            }
        }
        self.push("E");

        Ok(self)
    }
}
