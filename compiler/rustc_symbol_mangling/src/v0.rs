use std::fmt::Write;
use std::hash::Hasher;
use std::iter;
use std::ops::Range;

use rustc_abi::{ExternAbi, Integer};
use rustc_data_structures::base_n::ToBaseN;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_hashes::Hash64;
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::bug;
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::print::{Print, PrintError, Printer};
use rustc_middle::ty::{
    self, FloatTy, GenericArg, GenericArgKind, Instance, IntTy, ReifyReason, Ty, TyCtxt,
    TypeVisitable, TypeVisitableExt, UintTy,
};
use rustc_span::kw;

pub(super) fn mangle<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    instantiating_crate: Option<CrateNum>,
) -> String {
    let def_id = instance.def_id();
    // FIXME(eddyb) this should ideally not be needed.
    let args = tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), instance.args);

    let prefix = "_R";
    let mut cx: SymbolMangler<'_> = SymbolMangler {
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
        ty::InstanceKind::ThreadLocalShim(_) => Some("tls"),
        ty::InstanceKind::VTableShim(_) => Some("vtable"),
        ty::InstanceKind::ReifyShim(_, None) => Some("reify"),
        ty::InstanceKind::ReifyShim(_, Some(ReifyReason::FnPtr)) => Some("reify_fnptr"),
        ty::InstanceKind::ReifyShim(_, Some(ReifyReason::Vtable)) => Some("reify_vtable"),

        // FIXME(async_closures): This shouldn't be needed when we fix
        // `Instance::ty`/`Instance::def_id`.
        ty::InstanceKind::ConstructCoroutineInClosureShim { receiver_by_ref: true, .. } => {
            Some("by_move")
        }
        ty::InstanceKind::ConstructCoroutineInClosureShim { receiver_by_ref: false, .. } => {
            Some("by_ref")
        }

        _ => None,
    };

    if let Some(shim_kind) = shim_kind {
        cx.path_append_ns(|cx| cx.print_def_path(def_id, args), 'S', 0, shim_kind).unwrap()
    } else {
        cx.print_def_path(def_id, args).unwrap()
    };
    if let Some(instantiating_crate) = instantiating_crate {
        cx.print_def_path(instantiating_crate.as_def_id(), &[]).unwrap();
    }
    std::mem::take(&mut cx.out)
}

pub fn mangle_internal_symbol<'tcx>(tcx: TyCtxt<'tcx>, item_name: &str) -> String {
    if item_name == "rust_eh_personality" {
        // rust_eh_personality must not be renamed as LLVM hard-codes the name
        return "rust_eh_personality".to_owned();
    } else if item_name == "__rust_no_alloc_shim_is_unstable" {
        // Temporary back compat hack to give people the chance to migrate to
        // include #[rustc_std_internal_symbol].
        return "__rust_no_alloc_shim_is_unstable".to_owned();
    }

    let prefix = "_R";
    let mut cx: SymbolMangler<'_> = SymbolMangler {
        tcx,
        start_offset: prefix.len(),
        paths: FxHashMap::default(),
        types: FxHashMap::default(),
        consts: FxHashMap::default(),
        binders: vec![],
        out: String::from(prefix),
    };

    cx.path_append_ns(
        |cx| {
            cx.push("C");
            cx.push_disambiguator({
                let mut hasher = StableHasher::new();
                // Incorporate the rustc version to ensure #[rustc_std_internal_symbol] functions
                // get a different symbol name depending on the rustc version.
                //
                // RUSTC_FORCE_RUSTC_VERSION is ignored here as otherwise different we would get an
                // abi incompatibility with the standard library.
                hasher.write(tcx.sess.cfg_version.as_bytes());

                let hash: Hash64 = hasher.finish();
                hash.as_u64()
            });
            cx.push_ident("__rustc");
            Ok(())
        },
        'v',
        0,
        item_name,
    )
    .unwrap();

    std::mem::take(&mut cx.out)
}

pub(super) fn mangle_typeid_for_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::ExistentialTraitRef<'tcx>,
) -> String {
    // FIXME(flip1995): See comment in `mangle_typeid_for_fnabi`.
    let mut cx = SymbolMangler {
        tcx,
        start_offset: 0,
        paths: FxHashMap::default(),
        types: FxHashMap::default(),
        consts: FxHashMap::default(),
        binders: vec![],
        out: String::new(),
    };
    cx.print_def_path(trait_ref.def_id, &[]).unwrap();
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
        push_integer_62(x, &mut self.out)
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
        push_ident(ident, &mut self.out)
    }

    fn path_append_ns(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        ns: char,
        disambiguator: u64,
        name: &str,
    ) -> Result<(), PrintError> {
        self.push("N");
        self.out.push(ns);
        print_prefix(self)?;
        self.push_disambiguator(disambiguator);
        self.push_ident(name);
        Ok(())
    }

    fn print_backref(&mut self, i: usize) -> Result<(), PrintError> {
        self.push("B");
        self.push_integer_62((i - self.start_offset) as u64);
        Ok(())
    }

    fn wrap_binder<T>(
        &mut self,
        value: &ty::Binder<'tcx, T>,
        print_value: impl FnOnce(&mut Self, &T) -> Result<(), PrintError>,
    ) -> Result<(), PrintError>
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
        print_value(self, value.as_ref().skip_binder())?;
        self.binders.pop();

        Ok(())
    }
}

impl<'tcx> Printer<'tcx> for SymbolMangler<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_def_path(
        &mut self,
        def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        if let Some(&i) = self.paths.get(&(def_id, args)) {
            return self.print_backref(i);
        }
        let start = self.out.len();

        self.default_print_def_path(def_id, args)?;

        // Only cache paths that do not refer to an enclosing
        // binder (which would change depending on context).
        if !args.iter().any(|k| k.has_escaping_bound_vars()) {
            self.paths.insert((def_id, args), start);
        }
        Ok(())
    }

    fn print_impl_path(
        &mut self,
        impl_def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        let key = self.tcx.def_key(impl_def_id);
        let parent_def_id = DefId { index: key.parent.unwrap(), ..impl_def_id };

        let self_ty = self.tcx.type_of(impl_def_id);
        let impl_trait_ref = self.tcx.impl_trait_ref(impl_def_id);
        let generics = self.tcx.generics_of(impl_def_id);
        // We have two cases to worry about here:
        // 1. We're printing a nested item inside of an impl item, like an inner
        // function inside of a method. Due to the way that def path printing works,
        // we'll render this something like `<Ty as Trait>::method::inner_fn`
        // but we have no substs for this impl since it's not really inheriting
        // generics from the outer item. We need to use the identity substs, and
        // to normalize we need to use the correct param-env too.
        // 2. We're mangling an item with identity substs. This seems to only happen
        // when generating coverage, since we try to generate coverage for unused
        // items too, and if something isn't monomorphized then we necessarily don't
        // have anything to substitute the instance with.
        // NOTE: We don't support mangling partially substituted but still polymorphic
        // instances, like `impl<A> Tr<A> for ()` where `A` is substituted w/ `(T,)`.
        let (typing_env, mut self_ty, mut impl_trait_ref) = if generics.count() > args.len()
            || &args[..generics.count()]
                == self
                    .tcx
                    .erase_regions(ty::GenericArgs::identity_for_item(self.tcx, impl_def_id))
                    .as_slice()
        {
            (
                ty::TypingEnv::post_analysis(self.tcx, impl_def_id),
                self_ty.instantiate_identity(),
                impl_trait_ref.map(|impl_trait_ref| impl_trait_ref.instantiate_identity()),
            )
        } else {
            assert!(
                !args.has_non_region_param(),
                "should not be mangling partially substituted \
                polymorphic instance: {impl_def_id:?} {args:?}"
            );
            (
                ty::TypingEnv::fully_monomorphized(),
                self_ty.instantiate(self.tcx, args),
                impl_trait_ref.map(|impl_trait_ref| impl_trait_ref.instantiate(self.tcx, args)),
            )
        };

        match &mut impl_trait_ref {
            Some(impl_trait_ref) => {
                assert_eq!(impl_trait_ref.self_ty(), self_ty);
                *impl_trait_ref = self.tcx.normalize_erasing_regions(typing_env, *impl_trait_ref);
                self_ty = impl_trait_ref.self_ty();
            }
            None => {
                self_ty = self.tcx.normalize_erasing_regions(typing_env, self_ty);
            }
        }

        self.push(match impl_trait_ref {
            Some(_) => "X",
            None => "M",
        });

        // Encode impl generic params if the generic parameters contain non-region parameters
        // and this isn't an inherent impl.
        if impl_trait_ref.is_some() && args.iter().any(|a| a.has_non_region_param()) {
            self.path_generic_args(
                |this| {
                    this.path_append_ns(
                        |cx| cx.print_def_path(parent_def_id, &[]),
                        'I',
                        key.disambiguated_data.disambiguator as u64,
                        "",
                    )
                },
                args,
            )?;
        } else {
            self.push_disambiguator(key.disambiguated_data.disambiguator as u64);
            self.print_def_path(parent_def_id, &[])?;
        }

        self_ty.print(self)?;

        if let Some(trait_ref) = impl_trait_ref {
            self.print_def_path(trait_ref.def_id, trait_ref.args)?;
        }

        Ok(())
    }

    fn print_region(&mut self, region: ty::Region<'_>) -> Result<(), PrintError> {
        let i = match *region {
            // Erased lifetimes use the index 0, for a
            // shorter mangling of `L_`.
            ty::ReErased => 0,

            // Bound lifetimes use indices starting at 1,
            // see `BinderLevel` for more details.
            ty::ReBound(debruijn, ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon }) => {
                let binder = &self.binders[self.binders.len() - 1 - debruijn.index()];
                let depth = binder.lifetime_depths.start + var.as_u32();

                1 + (self.binders.last().unwrap().lifetime_depths.end - 1 - depth)
            }

            _ => bug!("symbol_names: non-erased region `{:?}`", region),
        };
        self.push("L");
        self.push_integer_62(i as u64);
        Ok(())
    }

    fn print_type(&mut self, ty: Ty<'tcx>) -> Result<(), PrintError> {
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
            ty::Float(FloatTy::F16) => "C3f16",
            ty::Float(FloatTy::F32) => "f",
            ty::Float(FloatTy::F64) => "d",
            ty::Float(FloatTy::F128) => "C4f128",
            ty::Never => "z",

            // Should only be encountered within the identity-substituted
            // impl header of an item nested within an impl item.
            ty::Param(_) => "p",

            ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => bug!(),

            _ => "",
        };
        if !basic_type.is_empty() {
            self.push(basic_type);
            return Ok(());
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
                    r.print(self)?;
                }
                ty.print(self)?;
            }

            ty::RawPtr(ty, mutbl) => {
                self.push(match mutbl {
                    hir::Mutability::Not => "P",
                    hir::Mutability::Mut => "O",
                });
                ty.print(self)?;
            }

            ty::Pat(ty, pat) => match *pat {
                ty::PatternKind::Range { start, end } => {
                    let consts = [start, end];
                    // HACK: Represent as tuple until we have something better.
                    // HACK: constants are used in arrays, even if the types don't match.
                    self.push("T");
                    ty.print(self)?;
                    for ct in consts {
                        Ty::new_array_with_const_len(self.tcx, self.tcx.types.unit, ct)
                            .print(self)?;
                    }
                    self.push("E");
                }
            },

            ty::Array(ty, len) => {
                self.push("A");
                ty.print(self)?;
                self.print_const(len)?;
            }
            ty::Slice(ty) => {
                self.push("S");
                ty.print(self)?;
            }

            ty::Tuple(tys) => {
                self.push("T");
                for ty in tys.iter() {
                    ty.print(self)?;
                }
                self.push("E");
            }

            // Mangle all nominal types as paths.
            ty::Adt(ty::AdtDef(Interned(&ty::AdtDefData { did: def_id, .. }, _)), args)
            | ty::FnDef(def_id, args)
            | ty::Closure(def_id, args)
            | ty::CoroutineClosure(def_id, args)
            | ty::Coroutine(def_id, args) => {
                self.print_def_path(def_id, args)?;
            }

            // We may still encounter projections here due to the printing
            // logic sometimes passing identity-substituted impl headers.
            ty::Alias(ty::Projection, ty::AliasTy { def_id, args, .. }) => {
                self.print_def_path(def_id, args)?;
            }

            ty::Foreign(def_id) => {
                self.print_def_path(def_id, &[])?;
            }

            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                self.push("F");
                self.wrap_binder(&sig, |cx, sig| {
                    if sig.safety.is_unsafe() {
                        cx.push("U");
                    }
                    match sig.abi {
                        ExternAbi::Rust => {}
                        ExternAbi::C { unwind: false } => cx.push("KC"),
                        abi => {
                            cx.push("K");
                            let name = abi.as_str();
                            if name.contains('-') {
                                cx.push_ident(&name.replace('-', "_"));
                            } else {
                                cx.push_ident(name);
                            }
                        }
                    }
                    for &ty in sig.inputs() {
                        ty.print(cx)?;
                    }
                    if sig.c_variadic {
                        cx.push("v");
                    }
                    cx.push("E");
                    sig.output().print(cx)
                })?;
            }

            // FIXME(unsafe_binder):
            ty::UnsafeBinder(..) => todo!(),

            ty::Dynamic(predicates, r, kind) => {
                self.push(match kind {
                    ty::Dyn => "D",
                    // FIXME(dyn-star): need to update v0 mangling docs
                    ty::DynStar => "D*",
                });
                self.print_dyn_existential(predicates)?;
                r.print(self)?;
            }

            ty::Alias(..) => bug!("symbol_names: unexpected alias"),
            ty::CoroutineWitness(..) => bug!("symbol_names: unexpected `CoroutineWitness`"),
        }

        // Only cache types that do not refer to an enclosing
        // binder (which would change depending on context).
        if !ty.has_escaping_bound_vars() {
            self.types.insert(ty, start);
        }
        Ok(())
    }

    fn print_dyn_existential(
        &mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<(), PrintError> {
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
        self.wrap_binder(&predicates[0], |cx, _| {
            for predicate in predicates.iter() {
                // It would be nice to be able to validate bound vars here, but
                // projections can actually include bound vars from super traits
                // because of HRTBs (only in the `Self` type). Also, auto traits
                // could have different bound vars *anyways*.
                match predicate.as_ref().skip_binder() {
                    ty::ExistentialPredicate::Trait(trait_ref) => {
                        // Use a type that can't appear in defaults of type parameters.
                        let dummy_self = Ty::new_fresh(cx.tcx, 0);
                        let trait_ref = trait_ref.with_self_ty(cx.tcx, dummy_self);
                        cx.print_def_path(trait_ref.def_id, trait_ref.args)?;
                    }
                    ty::ExistentialPredicate::Projection(projection) => {
                        let name = cx.tcx.associated_item(projection.def_id).name;
                        cx.push("p");
                        cx.push_ident(name.as_str());
                        match projection.term.unpack() {
                            ty::TermKind::Ty(ty) => ty.print(cx),
                            ty::TermKind::Const(c) => c.print(cx),
                        }?;
                    }
                    ty::ExistentialPredicate::AutoTrait(def_id) => {
                        cx.print_def_path(*def_id, &[])?;
                    }
                }
            }
            Ok(())
        })?;

        self.push("E");
        Ok(())
    }

    fn print_const(&mut self, ct: ty::Const<'tcx>) -> Result<(), PrintError> {
        // We only mangle a typed value if the const can be evaluated.
        let cv = match ct.kind() {
            ty::ConstKind::Value(cv) => cv,

            // Should only be encountered within the identity-substituted
            // impl header of an item nested within an impl item.
            ty::ConstKind::Param(_) => {
                // Never cached (single-character).
                self.push("p");
                return Ok(());
            }

            // We may still encounter unevaluated consts due to the printing
            // logic sometimes passing identity-substituted impl headers.
            ty::ConstKind::Unevaluated(ty::UnevaluatedConst { def, args, .. }) => {
                return self.print_def_path(def, args);
            }

            ty::ConstKind::Expr(_)
            | ty::ConstKind::Infer(_)
            | ty::ConstKind::Bound(..)
            | ty::ConstKind::Placeholder(_)
            | ty::ConstKind::Error(_) => bug!(),
        };

        if let Some(&i) = self.consts.get(&ct) {
            self.print_backref(i)?;
            return Ok(());
        }

        let ty::Value { ty: ct_ty, valtree } = cv;
        let start = self.out.len();

        match ct_ty.kind() {
            ty::Uint(_) | ty::Int(_) | ty::Bool | ty::Char => {
                ct_ty.print(self)?;

                let mut bits = cv
                    .try_to_bits(self.tcx, ty::TypingEnv::fully_monomorphized())
                    .expect("expected const to be monomorphic");

                // Negative integer values are mangled using `n` as a "sign prefix".
                if let ty::Int(ity) = ct_ty.kind() {
                    let val =
                        Integer::from_int_ty(&self.tcx, *ity).size().sign_extend(bits) as i128;
                    if val < 0 {
                        self.push("n");
                    }
                    bits = val.unsigned_abs();
                }

                let _ = write!(self.out, "{bits:x}_");
            }

            // Handle `str` as partial support for unsized constants
            ty::Str => {
                let tcx = self.tcx();
                // HACK(jaic1): hide the `str` type behind a reference
                // for the following transformation from valtree to raw bytes
                let ref_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_static, ct_ty);
                let cv = ty::Value { ty: ref_ty, valtree };
                let slice = cv.try_to_raw_bytes(tcx).unwrap_or_else(|| {
                    bug!("expected to get raw bytes from valtree {:?} for type {:}", valtree, ct_ty)
                });
                let s = std::str::from_utf8(slice).expect("non utf8 str from MIR interpreter");

                // "e" for str as a basic type
                self.push("e");

                // FIXME(eddyb) use a specialized hex-encoding loop.
                for byte in s.bytes() {
                    let _ = write!(self.out, "{byte:02x}");
                }

                self.push("_");
            }

            // FIXME(valtrees): Remove the special case for `str`
            // here and fully support unsized constants.
            ty::Ref(_, _, mutbl) => {
                self.push(match mutbl {
                    hir::Mutability::Not => "R",
                    hir::Mutability::Mut => "Q",
                });

                let pointee_ty =
                    ct_ty.builtin_deref(true).expect("tried to dereference on non-ptr type");
                let dereferenced_const = ty::Const::new_value(self.tcx, valtree, pointee_ty);
                dereferenced_const.print(self)?;
            }

            ty::Array(..) | ty::Tuple(..) | ty::Adt(..) | ty::Slice(_) => {
                let contents = self.tcx.destructure_const(ct);
                let fields = contents.fields.iter().copied();

                let print_field_list = |this: &mut Self| {
                    for field in fields.clone() {
                        field.print(this)?;
                    }
                    this.push("E");
                    Ok(())
                };

                match *ct_ty.kind() {
                    ty::Array(..) | ty::Slice(_) => {
                        self.push("A");
                        print_field_list(self)?;
                    }
                    ty::Tuple(..) => {
                        self.push("T");
                        print_field_list(self)?;
                    }
                    ty::Adt(def, args) => {
                        let variant_idx =
                            contents.variant.expect("destructed const of adt without variant idx");
                        let variant_def = &def.variant(variant_idx);

                        self.push("V");
                        self.print_def_path(variant_def.def_id, args)?;

                        match variant_def.ctor_kind() {
                            Some(CtorKind::Const) => {
                                self.push("U");
                            }
                            Some(CtorKind::Fn) => {
                                self.push("T");
                                print_field_list(self)?;
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

                                    field.print(self)?;
                                }
                                self.push("E");
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => {
                bug!("symbol_names: unsupported constant of type `{}` ({:?})", ct_ty, ct);
            }
        }

        // Only cache consts that do not refer to an enclosing
        // binder (which would change depending on context).
        if !ct.has_escaping_bound_vars() {
            self.consts.insert(ct, start);
        }
        Ok(())
    }

    fn path_crate(&mut self, cnum: CrateNum) -> Result<(), PrintError> {
        self.push("C");
        let stable_crate_id = self.tcx.def_path_hash(cnum.as_def_id()).stable_crate_id();
        self.push_disambiguator(stable_crate_id.as_u64());
        let name = self.tcx.crate_name(cnum);
        self.push_ident(name.as_str());
        Ok(())
    }

    fn path_qualified(
        &mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        assert!(trait_ref.is_some());
        let trait_ref = trait_ref.unwrap();

        self.push("Y");
        self_ty.print(self)?;
        self.print_def_path(trait_ref.def_id, trait_ref.args)
    }

    fn path_append_impl(
        &mut self,
        _: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        _: &DisambiguatedDefPathData,
        _: Ty<'tcx>,
        _: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        // Inlined into `print_impl_path`
        unreachable!()
    }

    fn path_append(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<(), PrintError> {
        let ns = match disambiguated_data.data {
            // Extern block segments can be skipped, names from extern blocks
            // are effectively living in their parent modules.
            DefPathData::ForeignMod => return print_prefix(self),

            // Uppercase categories are more stable than lowercase ones.
            DefPathData::TypeNs(_) => 't',
            DefPathData::ValueNs(_) => 'v',
            DefPathData::Closure => 'C',
            DefPathData::Ctor => 'c',
            DefPathData::AnonConst => 'k',
            DefPathData::OpaqueTy => 'i',
            DefPathData::SyntheticCoroutineBody => 's',

            // These should never show up as `path_append` arguments.
            DefPathData::CrateRoot
            | DefPathData::Use
            | DefPathData::GlobalAsm
            | DefPathData::Impl
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
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
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
        print_prefix(self)?;
        for arg in args {
            match arg.unpack() {
                GenericArgKind::Lifetime(lt) => {
                    lt.print(self)?;
                }
                GenericArgKind::Type(ty) => {
                    ty.print(self)?;
                }
                GenericArgKind::Const(c) => {
                    self.push("K");
                    c.print(self)?;
                }
            }
        }
        self.push("E");

        Ok(())
    }
}
/// Push a `_`-terminated base 62 integer, using the format
/// specified in the RFC as `<base-62-number>`, that is:
/// * `x = 0` is encoded as just the `"_"` terminator
/// * `x > 0` is encoded as `x - 1` in base 62, followed by `"_"`,
///   e.g. `1` becomes `"0_"`, `62` becomes `"Z_"`, etc.
pub(crate) fn push_integer_62(x: u64, output: &mut String) {
    if let Some(x) = x.checked_sub(1) {
        output.push_str(&x.to_base(62));
    }
    output.push('_');
}

pub(crate) fn encode_integer_62(x: u64) -> String {
    let mut output = String::new();
    push_integer_62(x, &mut output);
    output
}

pub(crate) fn push_ident(ident: &str, output: &mut String) {
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
        output.push('u');

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

    let _ = write!(output, "{}", ident.len());

    // Write a separating `_` if necessary (leading digit or `_`).
    if let Some('_' | '0'..='9') = ident.chars().next() {
        output.push('_');
    }

    output.push_str(ident);
}
