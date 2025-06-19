use hir::def_id::DefId;
use rustc_abi::Integer::{I8, I32};
use rustc_abi::Primitive::{self, Float, Int, Pointer};
use rustc_abi::{
    AddressSpace, BackendRepr, FIRST_VARIANT, FieldIdx, FieldsShape, HasDataLayout, Layout,
    LayoutCalculatorError, LayoutData, Niche, ReprOptions, Scalar, Size, StructKind, TagEncoding,
    VariantIdx, Variants, WrappingRange,
};
use rustc_hashes::Hash64;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{
    FloatExt, HasTyCtxt, IntegerExt, LayoutCx, LayoutError, LayoutOf, TyAndLayout,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, AdtDef, CoroutineArgsExt, EarlyBinder, PseudoCanonicalInput, Ty, TyCtxt, TypeVisitableExt,
};
use rustc_session::{DataTypeKind, FieldInfo, FieldKind, SizeKind, VariantInfo};
use rustc_span::{Symbol, sym};
use tracing::{debug, instrument};
use {rustc_abi as abi, rustc_hir as hir};

use crate::errors::{NonPrimitiveSimdType, OversizedSimdType, ZeroLengthSimdType};

mod invariant;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { layout_of, ..*providers };
}

#[instrument(skip(tcx, query), level = "debug")]
fn layout_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> Result<TyAndLayout<'tcx>, &'tcx LayoutError<'tcx>> {
    let PseudoCanonicalInput { typing_env, value: ty } = query;
    debug!(?ty);

    // Optimization: We convert to TypingMode::PostAnalysis and convert opaque types in
    // the where bounds to their hidden types. This reduces overall uncached invocations
    // of `layout_of` and is thus a small performance improvement.
    let typing_env = typing_env.with_post_analysis_normalized(tcx);
    let unnormalized_ty = ty;

    // FIXME: We might want to have two different versions of `layout_of`:
    // One that can be called after typecheck has completed and can use
    // `normalize_erasing_regions` here and another one that can be called
    // before typecheck has completed and uses `try_normalize_erasing_regions`.
    let ty = match tcx.try_normalize_erasing_regions(typing_env, ty) {
        Ok(t) => t,
        Err(normalization_error) => {
            return Err(tcx
                .arena
                .alloc(LayoutError::NormalizationFailure(ty, normalization_error)));
        }
    };

    if ty != unnormalized_ty {
        // Ensure this layout is also cached for the normalized type.
        return tcx.layout_of(typing_env.as_query_input(ty));
    }

    let cx = LayoutCx::new(tcx, typing_env);

    let layout = layout_of_uncached(&cx, ty)?;
    let layout = TyAndLayout { ty, layout };

    // If we are running with `-Zprint-type-sizes`, maybe record layouts
    // for dumping later.
    if cx.tcx().sess.opts.unstable_opts.print_type_sizes {
        record_layout_for_printing(&cx, layout);
    }

    invariant::layout_sanity_check(&cx, &layout);

    Ok(layout)
}

fn error<'tcx>(cx: &LayoutCx<'tcx>, err: LayoutError<'tcx>) -> &'tcx LayoutError<'tcx> {
    cx.tcx().arena.alloc(err)
}

fn map_error<'tcx>(
    cx: &LayoutCx<'tcx>,
    ty: Ty<'tcx>,
    err: LayoutCalculatorError<TyAndLayout<'tcx>>,
) -> &'tcx LayoutError<'tcx> {
    let err = match err {
        LayoutCalculatorError::SizeOverflow => {
            // This is sometimes not a compile error in `check` builds.
            // See `tests/ui/limits/huge-enum.rs` for an example.
            LayoutError::SizeOverflow(ty)
        }
        LayoutCalculatorError::UnexpectedUnsized(field) => {
            // This is sometimes not a compile error if there are trivially false where clauses.
            // See `tests/ui/layout/trivial-bounds-sized.rs` for an example.
            assert!(field.layout.is_unsized(), "invalid layout error {err:#?}");
            if cx.typing_env.param_env.caller_bounds().is_empty() {
                cx.tcx().dcx().delayed_bug(format!(
                    "encountered unexpected unsized field in layout of {ty:?}: {field:#?}"
                ));
            }
            LayoutError::Unknown(ty)
        }
        LayoutCalculatorError::EmptyUnion => {
            // This is always a compile error.
            let guar =
                cx.tcx().dcx().delayed_bug(format!("computed layout of empty union: {ty:?}"));
            LayoutError::ReferencesError(guar)
        }
        LayoutCalculatorError::ReprConflict => {
            // packed enums are the only known trigger of this, but others might arise
            let guar = cx
                .tcx()
                .dcx()
                .delayed_bug(format!("computed impossible repr (packed enum?): {ty:?}"));
            LayoutError::ReferencesError(guar)
        }
        LayoutCalculatorError::ZeroLengthSimdType => {
            // Can't be caught in typeck if the array length is generic.
            cx.tcx().dcx().emit_fatal(ZeroLengthSimdType { ty })
        }
        LayoutCalculatorError::OversizedSimdType { max_lanes } => {
            // Can't be caught in typeck if the array length is generic.
            cx.tcx().dcx().emit_fatal(OversizedSimdType { ty, max_lanes })
        }
        LayoutCalculatorError::NonPrimitiveSimdType(field) => {
            // This error isn't caught in typeck, e.g., if
            // the element type of the vector is generic.
            cx.tcx().dcx().emit_fatal(NonPrimitiveSimdType { ty, e_ty: field.ty })
        }
    };
    error(cx, err)
}

fn extract_const_value<'tcx>(
    cx: &LayoutCx<'tcx>,
    ty: Ty<'tcx>,
    ct: ty::Const<'tcx>,
) -> Result<ty::Value<'tcx>, &'tcx LayoutError<'tcx>> {
    match ct.kind() {
        ty::ConstKind::Value(cv) => Ok(cv),
        ty::ConstKind::Param(_) | ty::ConstKind::Expr(_) => {
            if !ct.has_param() {
                bug!("failed to normalize const, but it is not generic: {ct:?}");
            }
            Err(error(cx, LayoutError::TooGeneric(ty)))
        }
        ty::ConstKind::Unevaluated(_) => {
            let err = if ct.has_param() {
                LayoutError::TooGeneric(ty)
            } else {
                // This case is reachable with unsatisfiable predicates and GCE (which will
                // cause anon consts to inherit the unsatisfiable predicates). For example
                // if we have an unsatisfiable `u8: Trait` bound, then it's not a compile
                // error to mention `[u8; <u8 as Trait>::CONST]`, but we can't compute its
                // layout.
                LayoutError::Unknown(ty)
            };
            Err(error(cx, err))
        }
        ty::ConstKind::Infer(_)
        | ty::ConstKind::Bound(..)
        | ty::ConstKind::Placeholder(_)
        | ty::ConstKind::Error(_) => {
            // `ty::ConstKind::Error` is handled at the top of `layout_of_uncached`
            // (via `ty.error_reported()`).
            bug!("layout_of: unexpected const: {ct:?}");
        }
    }
}

fn layout_of_uncached<'tcx>(
    cx: &LayoutCx<'tcx>,
    ty: Ty<'tcx>,
) -> Result<Layout<'tcx>, &'tcx LayoutError<'tcx>> {
    // Types that reference `ty::Error` pessimistically don't have a meaningful layout.
    // The only side-effect of this is possibly worse diagnostics in case the layout
    // was actually computable (like if the `ty::Error` showed up only in a `PhantomData`).
    if let Err(guar) = ty.error_reported() {
        return Err(error(cx, LayoutError::ReferencesError(guar)));
    }

    let tcx = cx.tcx();

    // layout of `async_drop_in_place<T>::{closure}` in case,
    // when T is a coroutine, contains this internal coroutine's ref

    let dl = cx.data_layout();
    let map_layout = |result: Result<_, _>| match result {
        Ok(layout) => Ok(tcx.mk_layout(layout)),
        Err(err) => Err(map_error(cx, ty, err)),
    };
    let scalar_unit = |value: Primitive| {
        let size = value.size(dl);
        assert!(size.bits() <= 128);
        Scalar::Initialized { value, valid_range: WrappingRange::full(size) }
    };
    let scalar = |value: Primitive| tcx.mk_layout(LayoutData::scalar(cx, scalar_unit(value)));

    let univariant = |tys: &[Ty<'tcx>], kind| {
        let fields = tys.iter().map(|ty| cx.layout_of(*ty)).try_collect::<IndexVec<_, _>>()?;
        let repr = ReprOptions::default();
        map_layout(cx.calc.univariant(&fields, &repr, kind))
    };
    debug_assert!(!ty.has_non_region_infer());

    Ok(match *ty.kind() {
        ty::Pat(ty, pat) => {
            let layout = cx.layout_of(ty)?.layout;
            let mut layout = LayoutData::clone(&layout.0);
            match *pat {
                ty::PatternKind::Range { start, end } => {
                    if let BackendRepr::Scalar(scalar) | BackendRepr::ScalarPair(scalar, _) =
                        &mut layout.backend_repr
                    {
                        scalar.valid_range_mut().start = extract_const_value(cx, ty, start)?
                            .try_to_bits(tcx, cx.typing_env)
                            .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?;

                        scalar.valid_range_mut().end = extract_const_value(cx, ty, end)?
                            .try_to_bits(tcx, cx.typing_env)
                            .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?;

                        // FIXME(pattern_types): create implied bounds from pattern types in signatures
                        // that require that the range end is >= the range start so that we can't hit
                        // this error anymore without first having hit a trait solver error.
                        // Very fuzzy on the details here, but pattern types are an internal impl detail,
                        // so we can just go with this for now
                        if scalar.is_signed() {
                            let range = scalar.valid_range_mut();
                            let start = layout.size.sign_extend(range.start);
                            let end = layout.size.sign_extend(range.end);
                            if end < start {
                                let guar = tcx.dcx().err(format!(
                                    "pattern type ranges cannot wrap: {start}..={end}"
                                ));

                                return Err(error(cx, LayoutError::ReferencesError(guar)));
                            }
                        } else {
                            let range = scalar.valid_range_mut();
                            if range.end < range.start {
                                let guar = tcx.dcx().err(format!(
                                    "pattern type ranges cannot wrap: {}..={}",
                                    range.start, range.end
                                ));

                                return Err(error(cx, LayoutError::ReferencesError(guar)));
                            }
                        };

                        let niche = Niche {
                            offset: Size::ZERO,
                            value: scalar.primitive(),
                            valid_range: scalar.valid_range(cx),
                        };

                        layout.largest_niche = Some(niche);
                    } else {
                        bug!("pattern type with range but not scalar layout: {ty:?}, {layout:?}")
                    }
                }
                ty::PatternKind::Or(variants) => match *variants[0] {
                    ty::PatternKind::Range { .. } => {
                        if let BackendRepr::Scalar(scalar) = &mut layout.backend_repr {
                            let variants: Result<Vec<_>, _> = variants
                                .iter()
                                .map(|pat| match *pat {
                                    ty::PatternKind::Range { start, end } => Ok((
                                        extract_const_value(cx, ty, start)
                                            .unwrap()
                                            .try_to_bits(tcx, cx.typing_env)
                                            .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?,
                                        extract_const_value(cx, ty, end)
                                            .unwrap()
                                            .try_to_bits(tcx, cx.typing_env)
                                            .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?,
                                    )),
                                    ty::PatternKind::Or(_) => {
                                        unreachable!("mixed or patterns are not allowed")
                                    }
                                })
                                .collect();
                            let mut variants = variants?;
                            if !scalar.is_signed() {
                                let guar = tcx.dcx().err(format!(
                                    "only signed integer base types are allowed for or-pattern pattern types at present"
                                ));

                                return Err(error(cx, LayoutError::ReferencesError(guar)));
                            }
                            variants.sort();
                            if variants.len() != 2 {
                                let guar = tcx
                                .dcx()
                                .err(format!("the only or-pattern types allowed are two range patterns that are directly connected at their overflow site"));

                                return Err(error(cx, LayoutError::ReferencesError(guar)));
                            }

                            // first is the one starting at the signed in range min
                            let mut first = variants[0];
                            let mut second = variants[1];
                            if second.0
                                == layout.size.truncate(layout.size.signed_int_min() as u128)
                            {
                                (second, first) = (first, second);
                            }

                            if layout.size.sign_extend(first.1) >= layout.size.sign_extend(second.0)
                            {
                                let guar = tcx.dcx().err(format!(
                                    "only non-overlapping pattern type ranges are allowed at present"
                                ));

                                return Err(error(cx, LayoutError::ReferencesError(guar)));
                            }
                            if layout.size.signed_int_max() as u128 != second.1 {
                                let guar = tcx.dcx().err(format!(
                                    "one pattern needs to end at `{ty}::MAX`, but was {} instead",
                                    second.1
                                ));

                                return Err(error(cx, LayoutError::ReferencesError(guar)));
                            }

                            // Now generate a wrapping range (which aren't allowed in surface syntax).
                            scalar.valid_range_mut().start = second.0;
                            scalar.valid_range_mut().end = first.1;

                            let niche = Niche {
                                offset: Size::ZERO,
                                value: scalar.primitive(),
                                valid_range: scalar.valid_range(cx),
                            };

                            layout.largest_niche = Some(niche);
                        } else {
                            bug!(
                                "pattern type with range but not scalar layout: {ty:?}, {layout:?}"
                            )
                        }
                    }
                    ty::PatternKind::Or(..) => bug!("patterns cannot have nested or patterns"),
                },
            }
            tcx.mk_layout(layout)
        }

        // Basic scalars.
        ty::Bool => tcx.mk_layout(LayoutData::scalar(
            cx,
            Scalar::Initialized {
                value: Int(I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            },
        )),
        ty::Char => tcx.mk_layout(LayoutData::scalar(
            cx,
            Scalar::Initialized {
                value: Int(I32, false),
                valid_range: WrappingRange { start: 0, end: 0x10FFFF },
            },
        )),
        ty::Int(ity) => scalar(Int(abi::Integer::from_int_ty(dl, ity), true)),
        ty::Uint(ity) => scalar(Int(abi::Integer::from_uint_ty(dl, ity), false)),
        ty::Float(fty) => scalar(Float(abi::Float::from_float_ty(fty))),
        ty::FnPtr(..) => {
            let mut ptr = scalar_unit(Pointer(dl.instruction_address_space));
            ptr.valid_range_mut().start = 1;
            tcx.mk_layout(LayoutData::scalar(cx, ptr))
        }

        // The never type.
        ty::Never => tcx.mk_layout(LayoutData::never_type(cx)),

        // Potentially-wide pointers.
        ty::Ref(_, pointee, _) | ty::RawPtr(pointee, _) => {
            let mut data_ptr = scalar_unit(Pointer(AddressSpace::DATA));
            if !ty.is_raw_ptr() {
                data_ptr.valid_range_mut().start = 1;
            }

            if pointee.is_sized(tcx, cx.typing_env) {
                return Ok(tcx.mk_layout(LayoutData::scalar(cx, data_ptr)));
            }

            let metadata = if let Some(metadata_def_id) = tcx.lang_items().metadata_type() {
                let pointee_metadata = Ty::new_projection(tcx, metadata_def_id, [pointee]);
                let metadata_ty =
                    match tcx.try_normalize_erasing_regions(cx.typing_env, pointee_metadata) {
                        Ok(metadata_ty) => metadata_ty,
                        Err(mut err) => {
                            // Usually `<Ty as Pointee>::Metadata` can't be normalized because
                            // its struct tail cannot be normalized either, so try to get a
                            // more descriptive layout error here, which will lead to less confusing
                            // diagnostics.
                            //
                            // We use the raw struct tail function here to get the first tail
                            // that is an alias, which is likely the cause of the normalization
                            // error.
                            match tcx.try_normalize_erasing_regions(
                                cx.typing_env,
                                tcx.struct_tail_raw(pointee, |ty| ty, || {}),
                            ) {
                                Ok(_) => {}
                                Err(better_err) => {
                                    err = better_err;
                                }
                            }
                            return Err(error(cx, LayoutError::NormalizationFailure(pointee, err)));
                        }
                    };

                let metadata_layout = cx.layout_of(metadata_ty)?;
                // If the metadata is a 1-zst, then the pointer is thin.
                if metadata_layout.is_1zst() {
                    return Ok(tcx.mk_layout(LayoutData::scalar(cx, data_ptr)));
                }

                let BackendRepr::Scalar(metadata) = metadata_layout.backend_repr else {
                    return Err(error(cx, LayoutError::Unknown(pointee)));
                };

                metadata
            } else {
                let unsized_part = tcx.struct_tail_for_codegen(pointee, cx.typing_env);

                match unsized_part.kind() {
                    ty::Foreign(..) => {
                        return Ok(tcx.mk_layout(LayoutData::scalar(cx, data_ptr)));
                    }
                    ty::Slice(_) | ty::Str => scalar_unit(Int(dl.ptr_sized_integer(), false)),
                    ty::Dynamic(..) => {
                        let mut vtable = scalar_unit(Pointer(AddressSpace::DATA));
                        vtable.valid_range_mut().start = 1;
                        vtable
                    }
                    _ => {
                        return Err(error(cx, LayoutError::Unknown(pointee)));
                    }
                }
            };

            // Effectively a (ptr, meta) tuple.
            tcx.mk_layout(LayoutData::scalar_pair(cx, data_ptr, metadata))
        }

        ty::Dynamic(_, _, ty::DynStar) => {
            let mut data = scalar_unit(Pointer(AddressSpace::DATA));
            data.valid_range_mut().start = 0;
            let mut vtable = scalar_unit(Pointer(AddressSpace::DATA));
            vtable.valid_range_mut().start = 1;
            tcx.mk_layout(LayoutData::scalar_pair(cx, data, vtable))
        }

        // Arrays and slices.
        ty::Array(element, count) => {
            let count = extract_const_value(cx, ty, count)?
                .try_to_target_usize(tcx)
                .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?;

            let element = cx.layout_of(element)?;
            map_layout(cx.calc.array_like(&element, Some(count)))?
        }
        ty::Slice(element) => {
            let element = cx.layout_of(element)?;
            map_layout(cx.calc.array_like(&element, None).map(|mut layout| {
                // a randomly chosen value to distinguish slices
                layout.randomization_seed = Hash64::new(0x2dcba99c39784102);
                layout
            }))?
        }
        ty::Str => {
            let element = scalar(Int(I8, false));
            map_layout(cx.calc.array_like(&element, None).map(|mut layout| {
                // another random value
                layout.randomization_seed = Hash64::new(0xc1325f37d127be22);
                layout
            }))?
        }

        // Odd unit types.
        ty::FnDef(..) | ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => {
            let sized = matches!(ty.kind(), ty::FnDef(..));
            tcx.mk_layout(LayoutData::unit(cx, sized))
        }

        ty::Coroutine(def_id, args) => {
            use rustc_middle::ty::layout::PrimitiveExt as _;

            let info = tcx.coroutine_layout(def_id, args)?;

            let local_layouts = info
                .field_tys
                .iter()
                .map(|local| {
                    let field_ty = EarlyBinder::bind(local.ty);
                    let uninit_ty = Ty::new_maybe_uninit(tcx, field_ty.instantiate(tcx, args));
                    cx.spanned_layout_of(uninit_ty, local.source_info.span)
                })
                .try_collect::<IndexVec<_, _>>()?;

            let prefix_layouts = args
                .as_coroutine()
                .prefix_tys()
                .iter()
                .map(|ty| cx.layout_of(ty))
                .try_collect::<IndexVec<_, _>>()?;

            let layout = cx
                .calc
                .coroutine(
                    &local_layouts,
                    prefix_layouts,
                    &info.variant_fields,
                    &info.storage_conflicts,
                    |tag| TyAndLayout {
                        ty: tag.primitive().to_ty(tcx),
                        layout: tcx.mk_layout(LayoutData::scalar(cx, tag)),
                    },
                )
                .map(|mut layout| {
                    // this is similar to how ReprOptions populates its field_shuffle_seed
                    layout.randomization_seed = tcx.def_path_hash(def_id).0.to_smaller_hash();
                    debug!("coroutine layout ({:?}): {:#?}", ty, layout);
                    layout
                });
            map_layout(layout)?
        }

        ty::Closure(_, args) => univariant(args.as_closure().upvar_tys(), StructKind::AlwaysSized)?,

        ty::CoroutineClosure(_, args) => {
            univariant(args.as_coroutine_closure().upvar_tys(), StructKind::AlwaysSized)?
        }

        ty::Tuple(tys) => {
            let kind =
                if tys.len() == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            univariant(tys, kind)?
        }

        // SIMD vector types.
        ty::Adt(def, args) if def.repr().simd() => {
            // Supported SIMD vectors are ADTs with a single array field:
            //
            // * #[repr(simd)] struct S([T; 4])
            //
            // where T is a primitive scalar (integer/float/pointer).
            let Some(ty::Array(e_ty, e_len)) = def
                .is_struct()
                .then(|| &def.variant(FIRST_VARIANT).fields)
                .filter(|fields| fields.len() == 1)
                .map(|fields| *fields[FieldIdx::ZERO].ty(tcx, args).kind())
            else {
                // Invalid SIMD types should have been caught by typeck by now.
                let guar = tcx.dcx().delayed_bug("#[repr(simd)] was applied to an invalid ADT");
                return Err(error(cx, LayoutError::ReferencesError(guar)));
            };

            let e_len = extract_const_value(cx, ty, e_len)?
                .try_to_target_usize(tcx)
                .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?;

            let e_ly = cx.layout_of(e_ty)?;

            map_layout(cx.calc.simd_type(e_ly, e_len, def.repr().packed()))?
        }

        // ADTs.
        ty::Adt(def, args) => {
            // Cache the field layouts.
            let variants = def
                .variants()
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|field| cx.layout_of(field.ty(tcx, args)))
                        .try_collect::<IndexVec<_, _>>()
                })
                .try_collect::<IndexVec<VariantIdx, _>>()?;

            if def.is_union() {
                if def.repr().pack.is_some() && def.repr().align.is_some() {
                    let guar = tcx.dcx().span_delayed_bug(
                        tcx.def_span(def.did()),
                        "union cannot be packed and aligned",
                    );
                    return Err(error(cx, LayoutError::ReferencesError(guar)));
                }

                return map_layout(cx.calc.layout_of_union(&def.repr(), &variants));
            }

            // UnsafeCell and UnsafePinned both disable niche optimizations
            let is_special_no_niche = def.is_unsafe_cell() || def.is_unsafe_pinned();

            let get_discriminant_type =
                |min, max| abi::Integer::repr_discr(tcx, ty, &def.repr(), min, max);

            let discriminants_iter = || {
                def.is_enum()
                    .then(|| def.discriminants(tcx).map(|(v, d)| (v, d.val as i128)))
                    .into_iter()
                    .flatten()
            };

            let dont_niche_optimize_enum = def.repr().inhibit_enum_layout_opt()
                || def
                    .variants()
                    .iter_enumerated()
                    .any(|(i, v)| v.discr != ty::VariantDiscr::Relative(i.as_u32()));

            let maybe_unsized = def.is_struct()
                && def.non_enum_variant().tail_opt().is_some_and(|last_field| {
                    let typing_env = ty::TypingEnv::post_analysis(tcx, def.did());
                    !tcx.type_of(last_field.did).instantiate_identity().is_sized(tcx, typing_env)
                });

            let layout = cx
                .calc
                .layout_of_struct_or_enum(
                    &def.repr(),
                    &variants,
                    def.is_enum(),
                    is_special_no_niche,
                    tcx.layout_scalar_valid_range(def.did()),
                    get_discriminant_type,
                    discriminants_iter(),
                    dont_niche_optimize_enum,
                    !maybe_unsized,
                )
                .map_err(|err| map_error(cx, ty, err))?;

            if !maybe_unsized && layout.is_unsized() {
                bug!("got unsized layout for type that cannot be unsized {ty:?}: {layout:#?}");
            }

            // If the struct tail is sized and can be unsized, check that unsizing doesn't move the fields around.
            if cfg!(debug_assertions)
                && maybe_unsized
                && def.non_enum_variant().tail().ty(tcx, args).is_sized(tcx, cx.typing_env)
            {
                let mut variants = variants;
                let tail_replacement = cx.layout_of(Ty::new_slice(tcx, tcx.types.u8)).unwrap();
                *variants[FIRST_VARIANT].raw.last_mut().unwrap() = tail_replacement;

                let Ok(unsized_layout) = cx.calc.layout_of_struct_or_enum(
                    &def.repr(),
                    &variants,
                    def.is_enum(),
                    is_special_no_niche,
                    tcx.layout_scalar_valid_range(def.did()),
                    get_discriminant_type,
                    discriminants_iter(),
                    dont_niche_optimize_enum,
                    !maybe_unsized,
                ) else {
                    bug!("failed to compute unsized layout of {ty:?}");
                };

                let FieldsShape::Arbitrary { offsets: sized_offsets, .. } = &layout.fields else {
                    bug!("unexpected FieldsShape for sized layout of {ty:?}: {:?}", layout.fields);
                };
                let FieldsShape::Arbitrary { offsets: unsized_offsets, .. } =
                    &unsized_layout.fields
                else {
                    bug!(
                        "unexpected FieldsShape for unsized layout of {ty:?}: {:?}",
                        unsized_layout.fields
                    );
                };

                let (sized_tail, sized_fields) = sized_offsets.raw.split_last().unwrap();
                let (unsized_tail, unsized_fields) = unsized_offsets.raw.split_last().unwrap();

                if sized_fields != unsized_fields {
                    bug!("unsizing {ty:?} changed field order!\n{layout:?}\n{unsized_layout:?}");
                }

                if sized_tail < unsized_tail {
                    bug!("unsizing {ty:?} moved tail backwards!\n{layout:?}\n{unsized_layout:?}");
                }
            }

            tcx.mk_layout(layout)
        }

        ty::UnsafeBinder(bound_ty) => {
            let ty = tcx.instantiate_bound_regions_with_erased(bound_ty.into());
            cx.layout_of(ty)?.layout
        }

        // Types with no meaningful known layout.
        ty::Param(_) | ty::Placeholder(..) => {
            return Err(error(cx, LayoutError::TooGeneric(ty)));
        }

        ty::Alias(..) => {
            // NOTE(eddyb) `layout_of` query should've normalized these away,
            // if that was possible, so there's no reason to try again here.
            let err = if ty.has_param() {
                LayoutError::TooGeneric(ty)
            } else {
                // This is only reachable with unsatisfiable predicates. For example, if we have
                // `u8: Iterator`, then we can't compute the layout of `<u8 as Iterator>::Item`.
                LayoutError::Unknown(ty)
            };
            return Err(error(cx, err));
        }

        ty::Bound(..) | ty::CoroutineWitness(..) | ty::Infer(_) | ty::Error(_) => {
            // `ty::Error` is handled at the top of this function.
            bug!("layout_of: unexpected type `{ty}`")
        }
    })
}

fn record_layout_for_printing<'tcx>(cx: &LayoutCx<'tcx>, layout: TyAndLayout<'tcx>) {
    // Ignore layouts that are done with non-empty environments or
    // non-monomorphic layouts, as the user only wants to see the stuff
    // resulting from the final codegen session.
    if layout.ty.has_non_region_param() || !cx.typing_env.param_env.caller_bounds().is_empty() {
        return;
    }

    // (delay format until we actually need it)
    let record = |kind, packed, opt_discr_size, variants| {
        let type_desc = with_no_trimmed_paths!(format!("{}", layout.ty));
        cx.tcx().sess.code_stats.record_type_size(
            kind,
            type_desc,
            layout.align.abi,
            layout.size,
            packed,
            opt_discr_size,
            variants,
        );
    };

    match *layout.ty.kind() {
        ty::Adt(adt_def, _) => {
            debug!("print-type-size t: `{:?}` process adt", layout.ty);
            let adt_kind = adt_def.adt_kind();
            let adt_packed = adt_def.repr().pack.is_some();
            let (variant_infos, opt_discr_size) = variant_info_for_adt(cx, layout, adt_def);
            record(adt_kind.into(), adt_packed, opt_discr_size, variant_infos);
        }

        ty::Coroutine(def_id, args) => {
            debug!("print-type-size t: `{:?}` record coroutine", layout.ty);
            // Coroutines always have a begin/poisoned/end state with additional suspend points
            let (variant_infos, opt_discr_size) =
                variant_info_for_coroutine(cx, layout, def_id, args);
            record(DataTypeKind::Coroutine, false, opt_discr_size, variant_infos);
        }

        ty::Closure(..) => {
            debug!("print-type-size t: `{:?}` record closure", layout.ty);
            record(DataTypeKind::Closure, false, None, vec![]);
        }

        _ => {
            debug!("print-type-size t: `{:?}` skip non-nominal", layout.ty);
        }
    };
}

fn variant_info_for_adt<'tcx>(
    cx: &LayoutCx<'tcx>,
    layout: TyAndLayout<'tcx>,
    adt_def: AdtDef<'tcx>,
) -> (Vec<VariantInfo>, Option<Size>) {
    let build_variant_info = |n: Option<Symbol>, flds: &[Symbol], layout: TyAndLayout<'tcx>| {
        let mut min_size = Size::ZERO;
        let field_info: Vec<_> = flds
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                let field_layout = layout.field(cx, i);
                let offset = layout.fields.offset(i);
                min_size = min_size.max(offset + field_layout.size);
                FieldInfo {
                    kind: FieldKind::AdtField,
                    name,
                    offset: offset.bytes(),
                    size: field_layout.size.bytes(),
                    align: field_layout.align.abi.bytes(),
                    type_name: None,
                }
            })
            .collect();

        VariantInfo {
            name: n,
            kind: if layout.is_unsized() { SizeKind::Min } else { SizeKind::Exact },
            align: layout.align.abi.bytes(),
            size: if min_size.bytes() == 0 { layout.size.bytes() } else { min_size.bytes() },
            fields: field_info,
        }
    };

    match layout.variants {
        Variants::Empty => (vec![], None),

        Variants::Single { index } => {
            debug!("print-type-size `{:#?}` variant {}", layout, adt_def.variant(index).name);
            let variant_def = &adt_def.variant(index);
            let fields: Vec<_> = variant_def.fields.iter().map(|f| f.name).collect();
            (vec![build_variant_info(Some(variant_def.name), &fields, layout)], None)
        }

        Variants::Multiple { tag, ref tag_encoding, .. } => {
            debug!(
                "print-type-size `{:#?}` adt general variants def {}",
                layout.ty,
                adt_def.variants().len()
            );
            let variant_infos: Vec<_> = adt_def
                .variants()
                .iter_enumerated()
                .map(|(i, variant_def)| {
                    let fields: Vec<_> = variant_def.fields.iter().map(|f| f.name).collect();
                    build_variant_info(Some(variant_def.name), &fields, layout.for_variant(cx, i))
                })
                .collect();

            (
                variant_infos,
                match tag_encoding {
                    TagEncoding::Direct => Some(tag.size(cx)),
                    _ => None,
                },
            )
        }
    }
}

fn variant_info_for_coroutine<'tcx>(
    cx: &LayoutCx<'tcx>,
    layout: TyAndLayout<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
) -> (Vec<VariantInfo>, Option<Size>) {
    use itertools::Itertools;

    let Variants::Multiple { tag, ref tag_encoding, tag_field, .. } = layout.variants else {
        return (vec![], None);
    };

    let coroutine = cx.tcx().coroutine_layout(def_id, args).unwrap();
    let upvar_names = cx.tcx().closure_saved_names_of_captured_variables(def_id);

    let mut upvars_size = Size::ZERO;
    let upvar_fields: Vec<_> = args
        .as_coroutine()
        .upvar_tys()
        .iter()
        .zip_eq(upvar_names)
        .enumerate()
        .map(|(field_idx, (_, name))| {
            let field_layout = layout.field(cx, field_idx);
            let offset = layout.fields.offset(field_idx);
            upvars_size = upvars_size.max(offset + field_layout.size);
            FieldInfo {
                kind: FieldKind::Upvar,
                name: *name,
                offset: offset.bytes(),
                size: field_layout.size.bytes(),
                align: field_layout.align.abi.bytes(),
                type_name: None,
            }
        })
        .collect();

    let mut variant_infos: Vec<_> = coroutine
        .variant_fields
        .iter_enumerated()
        .map(|(variant_idx, variant_def)| {
            let variant_layout = layout.for_variant(cx, variant_idx);
            let mut variant_size = Size::ZERO;
            let fields = variant_def
                .iter()
                .enumerate()
                .map(|(field_idx, local)| {
                    let field_name = coroutine.field_names[*local];
                    let field_layout = variant_layout.field(cx, field_idx);
                    let offset = variant_layout.fields.offset(field_idx);
                    // The struct is as large as the last field's end
                    variant_size = variant_size.max(offset + field_layout.size);
                    FieldInfo {
                        kind: FieldKind::CoroutineLocal,
                        name: field_name.unwrap_or_else(|| {
                            Symbol::intern(&format!(".coroutine_field{}", local.as_usize()))
                        }),
                        offset: offset.bytes(),
                        size: field_layout.size.bytes(),
                        align: field_layout.align.abi.bytes(),
                        // Include the type name if there is no field name, or if the name is the
                        // __awaitee placeholder symbol which means a child future being `.await`ed.
                        type_name: (field_name.is_none() || field_name == Some(sym::__awaitee))
                            .then(|| Symbol::intern(&field_layout.ty.to_string())),
                    }
                })
                .chain(upvar_fields.iter().copied())
                .collect();

            // If the variant has no state-specific fields, then it's the size of the upvars.
            if variant_size == Size::ZERO {
                variant_size = upvars_size;
            }

            // This `if` deserves some explanation.
            //
            // The layout code has a choice of where to place the discriminant of this coroutine.
            // If the discriminant of the coroutine is placed early in the layout (before the
            // variant's own fields), then it'll implicitly be counted towards the size of the
            // variant, since we use the maximum offset to calculate size.
            //    (side-note: I know this is a bit problematic given upvars placement, etc).
            //
            // This is important, since the layout printing code always subtracts this discriminant
            // size from the variant size if the struct is "enum"-like, so failing to account for it
            // will either lead to numerical underflow, or an underreported variant size...
            //
            // However, if the discriminant is placed past the end of the variant, then we need
            // to factor in the size of the discriminant manually. This really should be refactored
            // better, but this "works" for now.
            if layout.fields.offset(tag_field.as_usize()) >= variant_size {
                variant_size += match tag_encoding {
                    TagEncoding::Direct => tag.size(cx),
                    _ => Size::ZERO,
                };
            }

            VariantInfo {
                name: Some(Symbol::intern(&ty::CoroutineArgs::variant_name(variant_idx))),
                kind: SizeKind::Exact,
                size: variant_size.bytes(),
                align: variant_layout.align.abi.bytes(),
                fields,
            }
        })
        .collect();

    // The first three variants are hardcoded to be `UNRESUMED`, `RETURNED` and `POISONED`.
    // We will move the `RETURNED` and `POISONED` elements to the end so we
    // are left with a sorting order according to the coroutines yield points:
    // First `Unresumed`, then the `SuspendN` followed by `Returned` and `Panicked` (POISONED).
    let end_states = variant_infos.drain(1..=2);
    let end_states: Vec<_> = end_states.collect();
    variant_infos.extend(end_states);

    (
        variant_infos,
        match tag_encoding {
            TagEncoding::Direct => Some(tag.size(cx)),
            _ => None,
        },
    )
}
