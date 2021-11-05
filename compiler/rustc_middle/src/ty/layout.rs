use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::mir::{GeneratorLayout, GeneratorSavedLocal};
use crate::ty::subst::Subst;
use crate::ty::{self, subst::SubstsRef, ReprOptions, Ty, TyCtxt, TypeFoldable};
use rustc_ast as ast;
use rustc_attr as attr;
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::{Idx, IndexVec};
use rustc_session::{config::OptLevel, DataTypeKind, FieldInfo, SizeKind, VariantInfo};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::call::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, Conv, FnAbi, PassMode, Reg, RegKind,
};
use rustc_target::abi::*;
use rustc_target::spec::{abi::Abi as SpecAbi, HasTargetSpec, PanicStrategy, Target};

use std::cmp;
use std::fmt;
use std::iter;
use std::num::NonZeroUsize;
use std::ops::Bound;

use rand::{seq::SliceRandom, SeedableRng};
use rand_xoshiro::Xoshiro128StarStar;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers =
        ty::query::Providers { layout_of, fn_abi_of_fn_ptr, fn_abi_of_instance, ..*providers };
}

pub trait IntegerExt {
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>, signed: bool) -> Ty<'tcx>;
    fn from_attr<C: HasDataLayout>(cx: &C, ity: attr::IntType) -> Integer;
    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: ty::IntTy) -> Integer;
    fn from_uint_ty<C: HasDataLayout>(cx: &C, uty: ty::UintTy) -> Integer;
    fn repr_discr<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (Integer, bool);
}

impl IntegerExt for Integer {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>, signed: bool) -> Ty<'tcx> {
        match (*self, signed) {
            (I8, false) => tcx.types.u8,
            (I16, false) => tcx.types.u16,
            (I32, false) => tcx.types.u32,
            (I64, false) => tcx.types.u64,
            (I128, false) => tcx.types.u128,
            (I8, true) => tcx.types.i8,
            (I16, true) => tcx.types.i16,
            (I32, true) => tcx.types.i32,
            (I64, true) => tcx.types.i64,
            (I128, true) => tcx.types.i128,
        }
    }

    /// Gets the Integer type from an attr::IntType.
    fn from_attr<C: HasDataLayout>(cx: &C, ity: attr::IntType) -> Integer {
        let dl = cx.data_layout();

        match ity {
            attr::SignedInt(ast::IntTy::I8) | attr::UnsignedInt(ast::UintTy::U8) => I8,
            attr::SignedInt(ast::IntTy::I16) | attr::UnsignedInt(ast::UintTy::U16) => I16,
            attr::SignedInt(ast::IntTy::I32) | attr::UnsignedInt(ast::UintTy::U32) => I32,
            attr::SignedInt(ast::IntTy::I64) | attr::UnsignedInt(ast::UintTy::U64) => I64,
            attr::SignedInt(ast::IntTy::I128) | attr::UnsignedInt(ast::UintTy::U128) => I128,
            attr::SignedInt(ast::IntTy::Isize) | attr::UnsignedInt(ast::UintTy::Usize) => {
                dl.ptr_sized_integer()
            }
        }
    }

    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: ty::IntTy) -> Integer {
        match ity {
            ty::IntTy::I8 => I8,
            ty::IntTy::I16 => I16,
            ty::IntTy::I32 => I32,
            ty::IntTy::I64 => I64,
            ty::IntTy::I128 => I128,
            ty::IntTy::Isize => cx.data_layout().ptr_sized_integer(),
        }
    }
    fn from_uint_ty<C: HasDataLayout>(cx: &C, ity: ty::UintTy) -> Integer {
        match ity {
            ty::UintTy::U8 => I8,
            ty::UintTy::U16 => I16,
            ty::UintTy::U32 => I32,
            ty::UintTy::U64 => I64,
            ty::UintTy::U128 => I128,
            ty::UintTy::Usize => cx.data_layout().ptr_sized_integer(),
        }
    }

    /// Finds the appropriate Integer type and signedness for the given
    /// signed discriminant range and `#[repr]` attribute.
    /// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u128, max as u128));
        let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(&tcx, ity);
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                bug!(
                    "Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum `{}",
                    ty
                )
            }
            return (discr, ity.is_signed());
        }

        let at_least = if repr.c() {
            // This is usually I32, however it can be different on some platforms,
            // notably hexagon and arm-none/thumb-none
            tcx.data_layout().c_enum_min_size
        } else {
            // repr(Rust) enums try to be as small as possible
            I8
        };

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (cmp::max(unsigned_fit, at_least), false)
        } else {
            (cmp::max(signed_fit, at_least), true)
        }
    }
}

pub trait PrimitiveExt {
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
    fn to_int_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
}

impl PrimitiveExt for Primitive {
    #[inline]
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            Int(i, signed) => i.to_ty(tcx, signed),
            F32 => tcx.types.f32,
            F64 => tcx.types.f64,
            Pointer => tcx.mk_mut_ptr(tcx.mk_unit()),
        }
    }

    /// Return an *integer* type matching this primitive.
    /// Useful in particular when dealing with enum discriminants.
    #[inline]
    fn to_int_ty(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            Int(i, signed) => i.to_ty(tcx, signed),
            Pointer => tcx.types.usize,
            F32 | F64 => bug!("floats do not have an int type"),
        }
    }
}

/// The first half of a fat pointer.
///
/// - For a trait object, this is the address of the box.
/// - For a slice, this is the base address.
pub const FAT_PTR_ADDR: usize = 0;

/// The second half of a fat pointer.
///
/// - For a trait object, this is the address of the vtable.
/// - For a slice, this is the length.
pub const FAT_PTR_EXTRA: usize = 1;

/// The maximum supported number of lanes in a SIMD vector.
///
/// This value is selected based on backend support:
/// * LLVM does not appear to have a vector width limit.
/// * Cranelift stores the base-2 log of the lane count in a 4 bit integer.
pub const MAX_SIMD_LANES: u64 = 1 << 0xF;

#[derive(Copy, Clone, Debug, HashStable, TyEncodable, TyDecodable)]
pub enum LayoutError<'tcx> {
    Unknown(Ty<'tcx>),
    SizeOverflow(Ty<'tcx>),
}

impl<'tcx> fmt::Display for LayoutError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            LayoutError::Unknown(ty) => write!(f, "the type `{}` has an unknown layout", ty),
            LayoutError::SizeOverflow(ty) => {
                write!(f, "values of the type `{}` are too big for the current architecture", ty)
            }
        }
    }
}

fn layout_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Result<TyAndLayout<'tcx>, LayoutError<'tcx>> {
    ty::tls::with_related_context(tcx, move |icx| {
        let (param_env, ty) = query.into_parts();

        if !tcx.recursion_limit().value_within_limit(icx.layout_depth) {
            tcx.sess.fatal(&format!("overflow representing the type `{}`", ty));
        }

        // Update the ImplicitCtxt to increase the layout_depth
        let icx = ty::tls::ImplicitCtxt { layout_depth: icx.layout_depth + 1, ..icx.clone() };

        ty::tls::enter_context(&icx, |_| {
            let param_env = param_env.with_reveal_all_normalized(tcx);
            let unnormalized_ty = ty;
            let ty = tcx.normalize_erasing_regions(param_env, ty);
            if ty != unnormalized_ty {
                // Ensure this layout is also cached for the normalized type.
                return tcx.layout_of(param_env.and(ty));
            }

            let cx = LayoutCx { tcx, param_env };

            let layout = cx.layout_of_uncached(ty)?;
            let layout = TyAndLayout { ty, layout };

            cx.record_layout_for_printing(layout);

            // Type-level uninhabitedness should always imply ABI uninhabitedness.
            if tcx.conservative_is_privately_uninhabited(param_env.and(ty)) {
                assert!(layout.abi.is_uninhabited());
            }

            Ok(layout)
        })
    })
}

pub struct LayoutCx<'tcx, C> {
    pub tcx: C,
    pub param_env: ty::ParamEnv<'tcx>,
}

#[derive(Copy, Clone, Debug)]
enum StructKind {
    /// A tuple, closure, or univariant which cannot be coerced to unsized.
    AlwaysSized,
    /// A univariant, the last field of which may be coerced to unsized.
    MaybeUnsized,
    /// A univariant, but with a prefix of an arbitrary size & alignment (e.g., enum tag).
    Prefixed(Size, Align),
}

// Invert a bijective mapping, i.e. `invert(map)[y] = x` if `map[x] = y`.
// This is used to go between `memory_index` (source field order to memory order)
// and `inverse_memory_index` (memory order to source field order).
// See also `FieldsShape::Arbitrary::memory_index` for more details.
// FIXME(eddyb) build a better abstraction for permutations, if possible.
fn invert_mapping(map: &[u32]) -> Vec<u32> {
    let mut inverse = vec![0; map.len()];
    for i in 0..map.len() {
        inverse[map[i] as usize] = i as u32;
    }
    inverse
}

impl<'tcx> LayoutCx<'tcx, TyCtxt<'tcx>> {
    fn scalar_pair(&self, a: Scalar, b: Scalar) -> Layout {
        let dl = self.data_layout();
        let b_align = b.value.align(dl);
        let align = a.value.align(dl).max(b_align).max(dl.aggregate_align);
        let b_offset = a.value.size(dl).align_to(b_align.abi);
        let size = (b_offset + b.value.size(dl)).align_to(align.abi);

        // HACK(nox): We iter on `b` and then `a` because `max_by_key`
        // returns the last maximum.
        let largest_niche = Niche::from_scalar(dl, b_offset, b)
            .into_iter()
            .chain(Niche::from_scalar(dl, Size::ZERO, a))
            .max_by_key(|niche| niche.available(dl));

        Layout {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Arbitrary {
                offsets: vec![Size::ZERO, b_offset],
                memory_index: vec![0, 1],
            },
            abi: Abi::ScalarPair(a, b),
            largest_niche,
            align,
            size,
        }
    }

    fn univariant_uninterned(
        &self,
        ty: Ty<'tcx>,
        fields: &[TyAndLayout<'_>],
        repr: &ReprOptions,
        kind: StructKind,
    ) -> Result<Layout, LayoutError<'tcx>> {
        let dl = self.data_layout();
        let pack = repr.pack;
        if pack.is_some() && repr.align.is_some() {
            self.tcx.sess.delay_span_bug(DUMMY_SP, "struct cannot be packed and aligned");
            return Err(LayoutError::Unknown(ty));
        }

        let mut align = if pack.is_some() { dl.i8_align } else { dl.aggregate_align };

        let mut inverse_memory_index: Vec<u32> = (0..fields.len() as u32).collect();

        // `ReprOptions.layout_seed` is a deterministic seed that we can use to
        // randomize field ordering with
        let mut rng = Xoshiro128StarStar::seed_from_u64(repr.field_shuffle_seed);

        let optimize = !repr.inhibit_struct_field_reordering_opt();
        if optimize {
            let end =
                if let StructKind::MaybeUnsized = kind { fields.len() - 1 } else { fields.len() };
            let optimizing = &mut inverse_memory_index[..end];
            let field_align = |f: &TyAndLayout<'_>| {
                if let Some(pack) = pack { f.align.abi.min(pack) } else { f.align.abi }
            };

            // If `-Z randomize-layout` was enabled for the type definition we can shuffle
            // the field ordering to try and catch some code making assumptions about layouts
            // we don't guarantee
            if repr.can_randomize_type_layout() {
                // Shuffle the ordering of the fields
                optimizing.shuffle(&mut rng);

            // Otherwise we just leave things alone and actually optimize the type's fields
            } else {
                match kind {
                    StructKind::AlwaysSized | StructKind::MaybeUnsized => {
                        optimizing.sort_by_key(|&x| {
                            // Place ZSTs first to avoid "interesting offsets",
                            // especially with only one or two non-ZST fields.
                            let f = &fields[x as usize];
                            (!f.is_zst(), cmp::Reverse(field_align(f)))
                        });
                    }

                    StructKind::Prefixed(..) => {
                        // Sort in ascending alignment so that the layout stays optimal
                        // regardless of the prefix
                        optimizing.sort_by_key(|&x| field_align(&fields[x as usize]));
                    }
                }

                // FIXME(Kixiron): We can always shuffle fields within a given alignment class
                //                 regardless of the status of `-Z randomize-layout`
            }
        }

        // inverse_memory_index holds field indices by increasing memory offset.
        // That is, if field 5 has offset 0, the first element of inverse_memory_index is 5.
        // We now write field offsets to the corresponding offset slot;
        // field 5 with offset 0 puts 0 in offsets[5].
        // At the bottom of this function, we invert `inverse_memory_index` to
        // produce `memory_index` (see `invert_mapping`).

        let mut sized = true;
        let mut offsets = vec![Size::ZERO; fields.len()];
        let mut offset = Size::ZERO;
        let mut largest_niche = None;
        let mut largest_niche_available = 0;

        if let StructKind::Prefixed(prefix_size, prefix_align) = kind {
            let prefix_align =
                if let Some(pack) = pack { prefix_align.min(pack) } else { prefix_align };
            align = align.max(AbiAndPrefAlign::new(prefix_align));
            offset = prefix_size.align_to(prefix_align);
        }

        for &i in &inverse_memory_index {
            let field = fields[i as usize];
            if !sized {
                self.tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    &format!(
                        "univariant: field #{} of `{}` comes after unsized field",
                        offsets.len(),
                        ty
                    ),
                );
            }

            if field.is_unsized() {
                sized = false;
            }

            // Invariant: offset < dl.obj_size_bound() <= 1<<61
            let field_align = if let Some(pack) = pack {
                field.align.min(AbiAndPrefAlign::new(pack))
            } else {
                field.align
            };
            offset = offset.align_to(field_align.abi);
            align = align.max(field_align);

            debug!("univariant offset: {:?} field: {:#?}", offset, field);
            offsets[i as usize] = offset;

            if !repr.hide_niche() {
                if let Some(mut niche) = field.largest_niche {
                    let available = niche.available(dl);
                    if available > largest_niche_available {
                        largest_niche_available = available;
                        niche.offset += offset;
                        largest_niche = Some(niche);
                    }
                }
            }

            offset = offset.checked_add(field.size, dl).ok_or(LayoutError::SizeOverflow(ty))?;
        }

        if let Some(repr_align) = repr.align {
            align = align.max(AbiAndPrefAlign::new(repr_align));
        }

        debug!("univariant min_size: {:?}", offset);
        let min_size = offset;

        // As stated above, inverse_memory_index holds field indices by increasing offset.
        // This makes it an already-sorted view of the offsets vec.
        // To invert it, consider:
        // If field 5 has offset 0, offsets[0] is 5, and memory_index[5] should be 0.
        // Field 5 would be the first element, so memory_index is i:
        // Note: if we didn't optimize, it's already right.

        let memory_index =
            if optimize { invert_mapping(&inverse_memory_index) } else { inverse_memory_index };

        let size = min_size.align_to(align.abi);
        let mut abi = Abi::Aggregate { sized };

        // Unpack newtype ABIs and find scalar pairs.
        if sized && size.bytes() > 0 {
            // All other fields must be ZSTs.
            let mut non_zst_fields = fields.iter().enumerate().filter(|&(_, f)| !f.is_zst());

            match (non_zst_fields.next(), non_zst_fields.next(), non_zst_fields.next()) {
                // We have exactly one non-ZST field.
                (Some((i, field)), None, None) => {
                    // Field fills the struct and it has a scalar or scalar pair ABI.
                    if offsets[i].bytes() == 0 && align.abi == field.align.abi && size == field.size
                    {
                        match field.abi {
                            // For plain scalars, or vectors of them, we can't unpack
                            // newtypes for `#[repr(C)]`, as that affects C ABIs.
                            Abi::Scalar(_) | Abi::Vector { .. } if optimize => {
                                abi = field.abi;
                            }
                            // But scalar pairs are Rust-specific and get
                            // treated as aggregates by C ABIs anyway.
                            Abi::ScalarPair(..) => {
                                abi = field.abi;
                            }
                            _ => {}
                        }
                    }
                }

                // Two non-ZST fields, and they're both scalars.
                (
                    Some((i, &TyAndLayout { layout: &Layout { abi: Abi::Scalar(a), .. }, .. })),
                    Some((j, &TyAndLayout { layout: &Layout { abi: Abi::Scalar(b), .. }, .. })),
                    None,
                ) => {
                    // Order by the memory placement, not source order.
                    let ((i, a), (j, b)) =
                        if offsets[i] < offsets[j] { ((i, a), (j, b)) } else { ((j, b), (i, a)) };
                    let pair = self.scalar_pair(a, b);
                    let pair_offsets = match pair.fields {
                        FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                            assert_eq!(memory_index, &[0, 1]);
                            offsets
                        }
                        _ => bug!(),
                    };
                    if offsets[i] == pair_offsets[0]
                        && offsets[j] == pair_offsets[1]
                        && align == pair.align
                        && size == pair.size
                    {
                        // We can use `ScalarPair` only when it matches our
                        // already computed layout (including `#[repr(C)]`).
                        abi = pair.abi;
                    }
                }

                _ => {}
            }
        }

        if sized && fields.iter().any(|f| f.abi.is_uninhabited()) {
            abi = Abi::Uninhabited;
        }

        Ok(Layout {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Arbitrary { offsets, memory_index },
            abi,
            largest_niche,
            align,
            size,
        })
    }

    fn layout_of_uncached(&self, ty: Ty<'tcx>) -> Result<&'tcx Layout, LayoutError<'tcx>> {
        let tcx = self.tcx;
        let param_env = self.param_env;
        let dl = self.data_layout();
        let scalar_unit = |value: Primitive| {
            let size = value.size(dl);
            assert!(size.bits() <= 128);
            Scalar { value, valid_range: WrappingRange { start: 0, end: size.unsigned_int_max() } }
        };
        let scalar = |value: Primitive| tcx.intern_layout(Layout::scalar(self, scalar_unit(value)));

        let univariant = |fields: &[TyAndLayout<'_>], repr: &ReprOptions, kind| {
            Ok(tcx.intern_layout(self.univariant_uninterned(ty, fields, repr, kind)?))
        };
        debug_assert!(!ty.has_infer_types_or_consts());

        Ok(match *ty.kind() {
            // Basic scalars.
            ty::Bool => tcx.intern_layout(Layout::scalar(
                self,
                Scalar { value: Int(I8, false), valid_range: WrappingRange { start: 0, end: 1 } },
            )),
            ty::Char => tcx.intern_layout(Layout::scalar(
                self,
                Scalar {
                    value: Int(I32, false),
                    valid_range: WrappingRange { start: 0, end: 0x10FFFF },
                },
            )),
            ty::Int(ity) => scalar(Int(Integer::from_int_ty(dl, ity), true)),
            ty::Uint(ity) => scalar(Int(Integer::from_uint_ty(dl, ity), false)),
            ty::Float(fty) => scalar(match fty {
                ty::FloatTy::F32 => F32,
                ty::FloatTy::F64 => F64,
            }),
            ty::FnPtr(_) => {
                let mut ptr = scalar_unit(Pointer);
                ptr.valid_range = ptr.valid_range.with_start(1);
                tcx.intern_layout(Layout::scalar(self, ptr))
            }

            // The never type.
            ty::Never => tcx.intern_layout(Layout {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields: FieldsShape::Primitive,
                abi: Abi::Uninhabited,
                largest_niche: None,
                align: dl.i8_align,
                size: Size::ZERO,
            }),

            // Potentially-wide pointers.
            ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                let mut data_ptr = scalar_unit(Pointer);
                if !ty.is_unsafe_ptr() {
                    data_ptr.valid_range = data_ptr.valid_range.with_start(1);
                }

                let pointee = tcx.normalize_erasing_regions(param_env, pointee);
                if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
                    return Ok(tcx.intern_layout(Layout::scalar(self, data_ptr)));
                }

                let unsized_part = tcx.struct_tail_erasing_lifetimes(pointee, param_env);
                let metadata = match unsized_part.kind() {
                    ty::Foreign(..) => {
                        return Ok(tcx.intern_layout(Layout::scalar(self, data_ptr)));
                    }
                    ty::Slice(_) | ty::Str => scalar_unit(Int(dl.ptr_sized_integer(), false)),
                    ty::Dynamic(..) => {
                        let mut vtable = scalar_unit(Pointer);
                        vtable.valid_range = vtable.valid_range.with_start(1);
                        vtable
                    }
                    _ => return Err(LayoutError::Unknown(unsized_part)),
                };

                // Effectively a (ptr, meta) tuple.
                tcx.intern_layout(self.scalar_pair(data_ptr, metadata))
            }

            // Arrays and slices.
            ty::Array(element, mut count) => {
                if count.has_projections() {
                    count = tcx.normalize_erasing_regions(param_env, count);
                    if count.has_projections() {
                        return Err(LayoutError::Unknown(ty));
                    }
                }

                let count = count.try_eval_usize(tcx, param_env).ok_or(LayoutError::Unknown(ty))?;
                let element = self.layout_of(element)?;
                let size =
                    element.size.checked_mul(count, dl).ok_or(LayoutError::SizeOverflow(ty))?;

                let abi =
                    if count != 0 && tcx.conservative_is_privately_uninhabited(param_env.and(ty)) {
                        Abi::Uninhabited
                    } else {
                        Abi::Aggregate { sized: true }
                    };

                let largest_niche = if count != 0 { element.largest_niche } else { None };

                tcx.intern_layout(Layout {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldsShape::Array { stride: element.size, count },
                    abi,
                    largest_niche,
                    align: element.align,
                    size,
                })
            }
            ty::Slice(element) => {
                let element = self.layout_of(element)?;
                tcx.intern_layout(Layout {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldsShape::Array { stride: element.size, count: 0 },
                    abi: Abi::Aggregate { sized: false },
                    largest_niche: None,
                    align: element.align,
                    size: Size::ZERO,
                })
            }
            ty::Str => tcx.intern_layout(Layout {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields: FieldsShape::Array { stride: Size::from_bytes(1), count: 0 },
                abi: Abi::Aggregate { sized: false },
                largest_niche: None,
                align: dl.i8_align,
                size: Size::ZERO,
            }),

            // Odd unit types.
            ty::FnDef(..) => univariant(&[], &ReprOptions::default(), StructKind::AlwaysSized)?,
            ty::Dynamic(..) | ty::Foreign(..) => {
                let mut unit = self.univariant_uninterned(
                    ty,
                    &[],
                    &ReprOptions::default(),
                    StructKind::AlwaysSized,
                )?;
                match unit.abi {
                    Abi::Aggregate { ref mut sized } => *sized = false,
                    _ => bug!(),
                }
                tcx.intern_layout(unit)
            }

            ty::Generator(def_id, substs, _) => self.generator_layout(ty, def_id, substs)?,

            ty::Closure(_, ref substs) => {
                let tys = substs.as_closure().upvar_tys();
                univariant(
                    &tys.map(|ty| self.layout_of(ty)).collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    StructKind::AlwaysSized,
                )?
            }

            ty::Tuple(tys) => {
                let kind =
                    if tys.len() == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

                univariant(
                    &tys.iter()
                        .map(|k| self.layout_of(k.expect_ty()))
                        .collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    kind,
                )?
            }

            // SIMD vector types.
            ty::Adt(def, substs) if def.repr.simd() => {
                if !def.is_struct() {
                    // Should have yielded E0517 by now.
                    tcx.sess.delay_span_bug(
                        DUMMY_SP,
                        "#[repr(simd)] was applied to an ADT that is not a struct",
                    );
                    return Err(LayoutError::Unknown(ty));
                }

                // Supported SIMD vectors are homogeneous ADTs with at least one field:
                //
                // * #[repr(simd)] struct S(T, T, T, T);
                // * #[repr(simd)] struct S { x: T, y: T, z: T, w: T }
                // * #[repr(simd)] struct S([T; 4])
                //
                // where T is a primitive scalar (integer/float/pointer).

                // SIMD vectors with zero fields are not supported.
                // (should be caught by typeck)
                if def.non_enum_variant().fields.is_empty() {
                    tcx.sess.fatal(&format!("monomorphising SIMD type `{}` of zero length", ty));
                }

                // Type of the first ADT field:
                let f0_ty = def.non_enum_variant().fields[0].ty(tcx, substs);

                // Heterogeneous SIMD vectors are not supported:
                // (should be caught by typeck)
                for fi in &def.non_enum_variant().fields {
                    if fi.ty(tcx, substs) != f0_ty {
                        tcx.sess.fatal(&format!("monomorphising heterogeneous SIMD type `{}`", ty));
                    }
                }

                // The element type and number of elements of the SIMD vector
                // are obtained from:
                //
                // * the element type and length of the single array field, if
                // the first field is of array type, or
                //
                // * the homogenous field type and the number of fields.
                let (e_ty, e_len, is_array) = if let ty::Array(e_ty, _) = f0_ty.kind() {
                    // First ADT field is an array:

                    // SIMD vectors with multiple array fields are not supported:
                    // (should be caught by typeck)
                    if def.non_enum_variant().fields.len() != 1 {
                        tcx.sess.fatal(&format!(
                            "monomorphising SIMD type `{}` with more than one array field",
                            ty
                        ));
                    }

                    // Extract the number of elements from the layout of the array field:
                    let Ok(TyAndLayout {
                        layout: Layout { fields: FieldsShape::Array { count, .. }, .. },
                        ..
                    }) = self.layout_of(f0_ty) else {
                        return Err(LayoutError::Unknown(ty));
                    };

                    (*e_ty, *count, true)
                } else {
                    // First ADT field is not an array:
                    (f0_ty, def.non_enum_variant().fields.len() as _, false)
                };

                // SIMD vectors of zero length are not supported.
                // Additionally, lengths are capped at 2^16 as a fixed maximum backends must
                // support.
                //
                // Can't be caught in typeck if the array length is generic.
                if e_len == 0 {
                    tcx.sess.fatal(&format!("monomorphising SIMD type `{}` of zero length", ty));
                } else if e_len > MAX_SIMD_LANES {
                    tcx.sess.fatal(&format!(
                        "monomorphising SIMD type `{}` of length greater than {}",
                        ty, MAX_SIMD_LANES,
                    ));
                }

                // Compute the ABI of the element type:
                let e_ly = self.layout_of(e_ty)?;
                let Abi::Scalar(e_abi) = e_ly.abi else {
                    // This error isn't caught in typeck, e.g., if
                    // the element type of the vector is generic.
                    tcx.sess.fatal(&format!(
                        "monomorphising SIMD type `{}` with a non-primitive-scalar \
                        (integer/float/pointer) element type `{}`",
                        ty, e_ty
                    ))
                };

                // Compute the size and alignment of the vector:
                let size = e_ly.size.checked_mul(e_len, dl).ok_or(LayoutError::SizeOverflow(ty))?;
                let align = dl.vector_align(size);
                let size = size.align_to(align.abi);

                // Compute the placement of the vector fields:
                let fields = if is_array {
                    FieldsShape::Arbitrary { offsets: vec![Size::ZERO], memory_index: vec![0] }
                } else {
                    FieldsShape::Array { stride: e_ly.size, count: e_len }
                };

                tcx.intern_layout(Layout {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields,
                    abi: Abi::Vector { element: e_abi, count: e_len },
                    largest_niche: e_ly.largest_niche,
                    size,
                    align,
                })
            }

            // ADTs.
            ty::Adt(def, substs) => {
                // Cache the field layouts.
                let variants = def
                    .variants
                    .iter()
                    .map(|v| {
                        v.fields
                            .iter()
                            .map(|field| self.layout_of(field.ty(tcx, substs)))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

                if def.is_union() {
                    if def.repr.pack.is_some() && def.repr.align.is_some() {
                        self.tcx.sess.delay_span_bug(
                            tcx.def_span(def.did),
                            "union cannot be packed and aligned",
                        );
                        return Err(LayoutError::Unknown(ty));
                    }

                    let mut align =
                        if def.repr.pack.is_some() { dl.i8_align } else { dl.aggregate_align };

                    if let Some(repr_align) = def.repr.align {
                        align = align.max(AbiAndPrefAlign::new(repr_align));
                    }

                    let optimize = !def.repr.inhibit_union_abi_opt();
                    let mut size = Size::ZERO;
                    let mut abi = Abi::Aggregate { sized: true };
                    let index = VariantIdx::new(0);
                    for field in &variants[index] {
                        assert!(!field.is_unsized());
                        align = align.max(field.align);

                        // If all non-ZST fields have the same ABI, forward this ABI
                        if optimize && !field.is_zst() {
                            // Normalize scalar_unit to the maximal valid range
                            let field_abi = match field.abi {
                                Abi::Scalar(x) => Abi::Scalar(scalar_unit(x.value)),
                                Abi::ScalarPair(x, y) => {
                                    Abi::ScalarPair(scalar_unit(x.value), scalar_unit(y.value))
                                }
                                Abi::Vector { element: x, count } => {
                                    Abi::Vector { element: scalar_unit(x.value), count }
                                }
                                Abi::Uninhabited | Abi::Aggregate { .. } => {
                                    Abi::Aggregate { sized: true }
                                }
                            };

                            if size == Size::ZERO {
                                // first non ZST: initialize 'abi'
                                abi = field_abi;
                            } else if abi != field_abi {
                                // different fields have different ABI: reset to Aggregate
                                abi = Abi::Aggregate { sized: true };
                            }
                        }

                        size = cmp::max(size, field.size);
                    }

                    if let Some(pack) = def.repr.pack {
                        align = align.min(AbiAndPrefAlign::new(pack));
                    }

                    return Ok(tcx.intern_layout(Layout {
                        variants: Variants::Single { index },
                        fields: FieldsShape::Union(
                            NonZeroUsize::new(variants[index].len())
                                .ok_or(LayoutError::Unknown(ty))?,
                        ),
                        abi,
                        largest_niche: None,
                        align,
                        size: size.align_to(align.abi),
                    }));
                }

                // A variant is absent if it's uninhabited and only has ZST fields.
                // Present uninhabited variants only require space for their fields,
                // but *not* an encoding of the discriminant (e.g., a tag value).
                // See issue #49298 for more details on the need to leave space
                // for non-ZST uninhabited data (mostly partial initialization).
                let absent = |fields: &[TyAndLayout<'_>]| {
                    let uninhabited = fields.iter().any(|f| f.abi.is_uninhabited());
                    let is_zst = fields.iter().all(|f| f.is_zst());
                    uninhabited && is_zst
                };
                let (present_first, present_second) = {
                    let mut present_variants = variants
                        .iter_enumerated()
                        .filter_map(|(i, v)| if absent(v) { None } else { Some(i) });
                    (present_variants.next(), present_variants.next())
                };
                let present_first = match present_first {
                    Some(present_first) => present_first,
                    // Uninhabited because it has no variants, or only absent ones.
                    None if def.is_enum() => {
                        return Ok(tcx.layout_of(param_env.and(tcx.types.never))?.layout);
                    }
                    // If it's a struct, still compute a layout so that we can still compute the
                    // field offsets.
                    None => VariantIdx::new(0),
                };

                let is_struct = !def.is_enum() ||
                    // Only one variant is present.
                    (present_second.is_none() &&
                    // Representation optimizations are allowed.
                    !def.repr.inhibit_enum_layout_opt());
                if is_struct {
                    // Struct, or univariant enum equivalent to a struct.
                    // (Typechecking will reject discriminant-sizing attrs.)

                    let v = present_first;
                    let kind = if def.is_enum() || variants[v].is_empty() {
                        StructKind::AlwaysSized
                    } else {
                        let param_env = tcx.param_env(def.did);
                        let last_field = def.variants[v].fields.last().unwrap();
                        let always_sized =
                            tcx.type_of(last_field.did).is_sized(tcx.at(DUMMY_SP), param_env);
                        if !always_sized {
                            StructKind::MaybeUnsized
                        } else {
                            StructKind::AlwaysSized
                        }
                    };

                    let mut st = self.univariant_uninterned(ty, &variants[v], &def.repr, kind)?;
                    st.variants = Variants::Single { index: v };
                    let (start, end) = self.tcx.layout_scalar_valid_range(def.did);
                    match st.abi {
                        Abi::Scalar(ref mut scalar) | Abi::ScalarPair(ref mut scalar, _) => {
                            // the asserts ensure that we are not using the
                            // `#[rustc_layout_scalar_valid_range(n)]`
                            // attribute to widen the range of anything as that would probably
                            // result in UB somewhere
                            // FIXME(eddyb) the asserts are probably not needed,
                            // as larger validity ranges would result in missed
                            // optimizations, *not* wrongly assuming the inner
                            // value is valid. e.g. unions enlarge validity ranges,
                            // because the values may be uninitialized.
                            if let Bound::Included(start) = start {
                                // FIXME(eddyb) this might be incorrect - it doesn't
                                // account for wrap-around (end < start) ranges.
                                assert!(scalar.valid_range.start <= start);
                                scalar.valid_range.start = start;
                            }
                            if let Bound::Included(end) = end {
                                // FIXME(eddyb) this might be incorrect - it doesn't
                                // account for wrap-around (end < start) ranges.
                                assert!(scalar.valid_range.end >= end);
                                scalar.valid_range.end = end;
                            }

                            // Update `largest_niche` if we have introduced a larger niche.
                            let niche = if def.repr.hide_niche() {
                                None
                            } else {
                                Niche::from_scalar(dl, Size::ZERO, *scalar)
                            };
                            if let Some(niche) = niche {
                                match st.largest_niche {
                                    Some(largest_niche) => {
                                        // Replace the existing niche even if they're equal,
                                        // because this one is at a lower offset.
                                        if largest_niche.available(dl) <= niche.available(dl) {
                                            st.largest_niche = Some(niche);
                                        }
                                    }
                                    None => st.largest_niche = Some(niche),
                                }
                            }
                        }
                        _ => assert!(
                            start == Bound::Unbounded && end == Bound::Unbounded,
                            "nonscalar layout for layout_scalar_valid_range type {:?}: {:#?}",
                            def,
                            st,
                        ),
                    }

                    return Ok(tcx.intern_layout(st));
                }

                // At this point, we have handled all unions and
                // structs. (We have also handled univariant enums
                // that allow representation optimization.)
                assert!(def.is_enum());

                // The current code for niche-filling relies on variant indices
                // instead of actual discriminants, so dataful enums with
                // explicit discriminants (RFC #2363) would misbehave.
                let no_explicit_discriminants = def
                    .variants
                    .iter_enumerated()
                    .all(|(i, v)| v.discr == ty::VariantDiscr::Relative(i.as_u32()));

                let mut niche_filling_layout = None;

                // Niche-filling enum optimization.
                if !def.repr.inhibit_enum_layout_opt() && no_explicit_discriminants {
                    let mut dataful_variant = None;
                    let mut niche_variants = VariantIdx::MAX..=VariantIdx::new(0);

                    // Find one non-ZST variant.
                    'variants: for (v, fields) in variants.iter_enumerated() {
                        if absent(fields) {
                            continue 'variants;
                        }
                        for f in fields {
                            if !f.is_zst() {
                                if dataful_variant.is_none() {
                                    dataful_variant = Some(v);
                                    continue 'variants;
                                } else {
                                    dataful_variant = None;
                                    break 'variants;
                                }
                            }
                        }
                        niche_variants = *niche_variants.start().min(&v)..=v;
                    }

                    if niche_variants.start() > niche_variants.end() {
                        dataful_variant = None;
                    }

                    if let Some(i) = dataful_variant {
                        let count = (niche_variants.end().as_u32()
                            - niche_variants.start().as_u32()
                            + 1) as u128;

                        // Find the field with the largest niche
                        let niche_candidate = variants[i]
                            .iter()
                            .enumerate()
                            .filter_map(|(j, field)| Some((j, field.largest_niche?)))
                            .max_by_key(|(_, niche)| niche.available(dl));

                        if let Some((field_index, niche, (niche_start, niche_scalar))) =
                            niche_candidate.and_then(|(field_index, niche)| {
                                Some((field_index, niche, niche.reserve(self, count)?))
                            })
                        {
                            let mut align = dl.aggregate_align;
                            let st = variants
                                .iter_enumerated()
                                .map(|(j, v)| {
                                    let mut st = self.univariant_uninterned(
                                        ty,
                                        v,
                                        &def.repr,
                                        StructKind::AlwaysSized,
                                    )?;
                                    st.variants = Variants::Single { index: j };

                                    align = align.max(st.align);

                                    Ok(st)
                                })
                                .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

                            let offset = st[i].fields.offset(field_index) + niche.offset;
                            let size = st[i].size;

                            let abi = if st.iter().all(|v| v.abi.is_uninhabited()) {
                                Abi::Uninhabited
                            } else {
                                match st[i].abi {
                                    Abi::Scalar(_) => Abi::Scalar(niche_scalar),
                                    Abi::ScalarPair(first, second) => {
                                        // We need to use scalar_unit to reset the
                                        // valid range to the maximal one for that
                                        // primitive, because only the niche is
                                        // guaranteed to be initialised, not the
                                        // other primitive.
                                        if offset.bytes() == 0 {
                                            Abi::ScalarPair(niche_scalar, scalar_unit(second.value))
                                        } else {
                                            Abi::ScalarPair(scalar_unit(first.value), niche_scalar)
                                        }
                                    }
                                    _ => Abi::Aggregate { sized: true },
                                }
                            };

                            let largest_niche = Niche::from_scalar(dl, offset, niche_scalar);

                            niche_filling_layout = Some(Layout {
                                variants: Variants::Multiple {
                                    tag: niche_scalar,
                                    tag_encoding: TagEncoding::Niche {
                                        dataful_variant: i,
                                        niche_variants,
                                        niche_start,
                                    },
                                    tag_field: 0,
                                    variants: st,
                                },
                                fields: FieldsShape::Arbitrary {
                                    offsets: vec![offset],
                                    memory_index: vec![0],
                                },
                                abi,
                                largest_niche,
                                size,
                                align,
                            });
                        }
                    }
                }

                let (mut min, mut max) = (i128::MAX, i128::MIN);
                let discr_type = def.repr.discr_type();
                let bits = Integer::from_attr(self, discr_type).size().bits();
                for (i, discr) in def.discriminants(tcx) {
                    if variants[i].iter().any(|f| f.abi.is_uninhabited()) {
                        continue;
                    }
                    let mut x = discr.val as i128;
                    if discr_type.is_signed() {
                        // sign extend the raw representation to be an i128
                        x = (x << (128 - bits)) >> (128 - bits);
                    }
                    if x < min {
                        min = x;
                    }
                    if x > max {
                        max = x;
                    }
                }
                // We might have no inhabited variants, so pretend there's at least one.
                if (min, max) == (i128::MAX, i128::MIN) {
                    min = 0;
                    max = 0;
                }
                assert!(min <= max, "discriminant range is {}...{}", min, max);
                let (min_ity, signed) = Integer::repr_discr(tcx, ty, &def.repr, min, max);

                let mut align = dl.aggregate_align;
                let mut size = Size::ZERO;

                // We're interested in the smallest alignment, so start large.
                let mut start_align = Align::from_bytes(256).unwrap();
                assert_eq!(Integer::for_align(dl, start_align), None);

                // repr(C) on an enum tells us to make a (tag, union) layout,
                // so we need to grow the prefix alignment to be at least
                // the alignment of the union. (This value is used both for
                // determining the alignment of the overall enum, and the
                // determining the alignment of the payload after the tag.)
                let mut prefix_align = min_ity.align(dl).abi;
                if def.repr.c() {
                    for fields in &variants {
                        for field in fields {
                            prefix_align = prefix_align.max(field.align.abi);
                        }
                    }
                }

                // Create the set of structs that represent each variant.
                let mut layout_variants = variants
                    .iter_enumerated()
                    .map(|(i, field_layouts)| {
                        let mut st = self.univariant_uninterned(
                            ty,
                            &field_layouts,
                            &def.repr,
                            StructKind::Prefixed(min_ity.size(), prefix_align),
                        )?;
                        st.variants = Variants::Single { index: i };
                        // Find the first field we can't move later
                        // to make room for a larger discriminant.
                        for field in
                            st.fields.index_by_increasing_offset().map(|j| field_layouts[j])
                        {
                            if !field.is_zst() || field.align.abi.bytes() != 1 {
                                start_align = start_align.min(field.align.abi);
                                break;
                            }
                        }
                        size = cmp::max(size, st.size);
                        align = align.max(st.align);
                        Ok(st)
                    })
                    .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

                // Align the maximum variant size to the largest alignment.
                size = size.align_to(align.abi);

                if size.bytes() >= dl.obj_size_bound() {
                    return Err(LayoutError::SizeOverflow(ty));
                }

                let typeck_ity = Integer::from_attr(dl, def.repr.discr_type());
                if typeck_ity < min_ity {
                    // It is a bug if Layout decided on a greater discriminant size than typeck for
                    // some reason at this point (based on values discriminant can take on). Mostly
                    // because this discriminant will be loaded, and then stored into variable of
                    // type calculated by typeck. Consider such case (a bug): typeck decided on
                    // byte-sized discriminant, but layout thinks we need a 16-bit to store all
                    // discriminant values. That would be a bug, because then, in codegen, in order
                    // to store this 16-bit discriminant into 8-bit sized temporary some of the
                    // space necessary to represent would have to be discarded (or layout is wrong
                    // on thinking it needs 16 bits)
                    bug!(
                        "layout decided on a larger discriminant type ({:?}) than typeck ({:?})",
                        min_ity,
                        typeck_ity
                    );
                    // However, it is fine to make discr type however large (as an optimisation)
                    // after this point – we’ll just truncate the value we load in codegen.
                }

                // Check to see if we should use a different type for the
                // discriminant. We can safely use a type with the same size
                // as the alignment of the first field of each variant.
                // We increase the size of the discriminant to avoid LLVM copying
                // padding when it doesn't need to. This normally causes unaligned
                // load/stores and excessive memcpy/memset operations. By using a
                // bigger integer size, LLVM can be sure about its contents and
                // won't be so conservative.

                // Use the initial field alignment
                let mut ity = if def.repr.c() || def.repr.int.is_some() {
                    min_ity
                } else {
                    Integer::for_align(dl, start_align).unwrap_or(min_ity)
                };

                // If the alignment is not larger than the chosen discriminant size,
                // don't use the alignment as the final size.
                if ity <= min_ity {
                    ity = min_ity;
                } else {
                    // Patch up the variants' first few fields.
                    let old_ity_size = min_ity.size();
                    let new_ity_size = ity.size();
                    for variant in &mut layout_variants {
                        match variant.fields {
                            FieldsShape::Arbitrary { ref mut offsets, .. } => {
                                for i in offsets {
                                    if *i <= old_ity_size {
                                        assert_eq!(*i, old_ity_size);
                                        *i = new_ity_size;
                                    }
                                }
                                // We might be making the struct larger.
                                if variant.size <= old_ity_size {
                                    variant.size = new_ity_size;
                                }
                            }
                            _ => bug!(),
                        }
                    }
                }

                let tag_mask = ity.size().unsigned_int_max();
                let tag = Scalar {
                    value: Int(ity, signed),
                    valid_range: WrappingRange {
                        start: (min as u128 & tag_mask),
                        end: (max as u128 & tag_mask),
                    },
                };
                let mut abi = Abi::Aggregate { sized: true };
                if tag.value.size(dl) == size {
                    abi = Abi::Scalar(tag);
                } else {
                    // Try to use a ScalarPair for all tagged enums.
                    let mut common_prim = None;
                    for (field_layouts, layout_variant) in iter::zip(&variants, &layout_variants) {
                        let offsets = match layout_variant.fields {
                            FieldsShape::Arbitrary { ref offsets, .. } => offsets,
                            _ => bug!(),
                        };
                        let mut fields =
                            iter::zip(field_layouts, offsets).filter(|p| !p.0.is_zst());
                        let (field, offset) = match (fields.next(), fields.next()) {
                            (None, None) => continue,
                            (Some(pair), None) => pair,
                            _ => {
                                common_prim = None;
                                break;
                            }
                        };
                        let prim = match field.abi {
                            Abi::Scalar(scalar) => scalar.value,
                            _ => {
                                common_prim = None;
                                break;
                            }
                        };
                        if let Some(pair) = common_prim {
                            // This is pretty conservative. We could go fancier
                            // by conflating things like i32 and u32, or even
                            // realising that (u8, u8) could just cohabit with
                            // u16 or even u32.
                            if pair != (prim, offset) {
                                common_prim = None;
                                break;
                            }
                        } else {
                            common_prim = Some((prim, offset));
                        }
                    }
                    if let Some((prim, offset)) = common_prim {
                        let pair = self.scalar_pair(tag, scalar_unit(prim));
                        let pair_offsets = match pair.fields {
                            FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                                assert_eq!(memory_index, &[0, 1]);
                                offsets
                            }
                            _ => bug!(),
                        };
                        if pair_offsets[0] == Size::ZERO
                            && pair_offsets[1] == *offset
                            && align == pair.align
                            && size == pair.size
                        {
                            // We can use `ScalarPair` only when it matches our
                            // already computed layout (including `#[repr(C)]`).
                            abi = pair.abi;
                        }
                    }
                }

                if layout_variants.iter().all(|v| v.abi.is_uninhabited()) {
                    abi = Abi::Uninhabited;
                }

                let largest_niche = Niche::from_scalar(dl, Size::ZERO, tag);

                let tagged_layout = Layout {
                    variants: Variants::Multiple {
                        tag,
                        tag_encoding: TagEncoding::Direct,
                        tag_field: 0,
                        variants: layout_variants,
                    },
                    fields: FieldsShape::Arbitrary {
                        offsets: vec![Size::ZERO],
                        memory_index: vec![0],
                    },
                    largest_niche,
                    abi,
                    align,
                    size,
                };

                let best_layout = match (tagged_layout, niche_filling_layout) {
                    (tagged_layout, Some(niche_filling_layout)) => {
                        // Pick the smaller layout; otherwise,
                        // pick the layout with the larger niche; otherwise,
                        // pick tagged as it has simpler codegen.
                        cmp::min_by_key(tagged_layout, niche_filling_layout, |layout| {
                            let niche_size = layout.largest_niche.map_or(0, |n| n.available(dl));
                            (layout.size, cmp::Reverse(niche_size))
                        })
                    }
                    (tagged_layout, None) => tagged_layout,
                };

                tcx.intern_layout(best_layout)
            }

            // Types with no meaningful known layout.
            ty::Projection(_) | ty::Opaque(..) => {
                // NOTE(eddyb) `layout_of` query should've normalized these away,
                // if that was possible, so there's no reason to try again here.
                return Err(LayoutError::Unknown(ty));
            }

            ty::Placeholder(..) | ty::GeneratorWitness(..) | ty::Infer(_) => {
                bug!("Layout::compute: unexpected type `{}`", ty)
            }

            ty::Bound(..) | ty::Param(_) | ty::Error(_) => {
                return Err(LayoutError::Unknown(ty));
            }
        })
    }
}

/// Overlap eligibility and variant assignment for each GeneratorSavedLocal.
#[derive(Clone, Debug, PartialEq)]
enum SavedLocalEligibility {
    Unassigned,
    Assigned(VariantIdx),
    // FIXME: Use newtype_index so we aren't wasting bytes
    Ineligible(Option<u32>),
}

// When laying out generators, we divide our saved local fields into two
// categories: overlap-eligible and overlap-ineligible.
//
// Those fields which are ineligible for overlap go in a "prefix" at the
// beginning of the layout, and always have space reserved for them.
//
// Overlap-eligible fields are only assigned to one variant, so we lay
// those fields out for each variant and put them right after the
// prefix.
//
// Finally, in the layout details, we point to the fields from the
// variants they are assigned to. It is possible for some fields to be
// included in multiple variants. No field ever "moves around" in the
// layout; its offset is always the same.
//
// Also included in the layout are the upvars and the discriminant.
// These are included as fields on the "outer" layout; they are not part
// of any variant.
impl<'tcx> LayoutCx<'tcx, TyCtxt<'tcx>> {
    /// Compute the eligibility and assignment of each local.
    fn generator_saved_local_eligibility(
        &self,
        info: &GeneratorLayout<'tcx>,
    ) -> (BitSet<GeneratorSavedLocal>, IndexVec<GeneratorSavedLocal, SavedLocalEligibility>) {
        use SavedLocalEligibility::*;

        let mut assignments: IndexVec<GeneratorSavedLocal, SavedLocalEligibility> =
            IndexVec::from_elem_n(Unassigned, info.field_tys.len());

        // The saved locals not eligible for overlap. These will get
        // "promoted" to the prefix of our generator.
        let mut ineligible_locals = BitSet::new_empty(info.field_tys.len());

        // Figure out which of our saved locals are fields in only
        // one variant. The rest are deemed ineligible for overlap.
        for (variant_index, fields) in info.variant_fields.iter_enumerated() {
            for local in fields {
                match assignments[*local] {
                    Unassigned => {
                        assignments[*local] = Assigned(variant_index);
                    }
                    Assigned(idx) => {
                        // We've already seen this local at another suspension
                        // point, so it is no longer a candidate.
                        trace!(
                            "removing local {:?} in >1 variant ({:?}, {:?})",
                            local,
                            variant_index,
                            idx
                        );
                        ineligible_locals.insert(*local);
                        assignments[*local] = Ineligible(None);
                    }
                    Ineligible(_) => {}
                }
            }
        }

        // Next, check every pair of eligible locals to see if they
        // conflict.
        for local_a in info.storage_conflicts.rows() {
            let conflicts_a = info.storage_conflicts.count(local_a);
            if ineligible_locals.contains(local_a) {
                continue;
            }

            for local_b in info.storage_conflicts.iter(local_a) {
                // local_a and local_b are storage live at the same time, therefore they
                // cannot overlap in the generator layout. The only way to guarantee
                // this is if they are in the same variant, or one is ineligible
                // (which means it is stored in every variant).
                if ineligible_locals.contains(local_b)
                    || assignments[local_a] == assignments[local_b]
                {
                    continue;
                }

                // If they conflict, we will choose one to make ineligible.
                // This is not always optimal; it's just a greedy heuristic that
                // seems to produce good results most of the time.
                let conflicts_b = info.storage_conflicts.count(local_b);
                let (remove, other) =
                    if conflicts_a > conflicts_b { (local_a, local_b) } else { (local_b, local_a) };
                ineligible_locals.insert(remove);
                assignments[remove] = Ineligible(None);
                trace!("removing local {:?} due to conflict with {:?}", remove, other);
            }
        }

        // Count the number of variants in use. If only one of them, then it is
        // impossible to overlap any locals in our layout. In this case it's
        // always better to make the remaining locals ineligible, so we can
        // lay them out with the other locals in the prefix and eliminate
        // unnecessary padding bytes.
        {
            let mut used_variants = BitSet::new_empty(info.variant_fields.len());
            for assignment in &assignments {
                if let Assigned(idx) = assignment {
                    used_variants.insert(*idx);
                }
            }
            if used_variants.count() < 2 {
                for assignment in assignments.iter_mut() {
                    *assignment = Ineligible(None);
                }
                ineligible_locals.insert_all();
            }
        }

        // Write down the order of our locals that will be promoted to the prefix.
        {
            for (idx, local) in ineligible_locals.iter().enumerate() {
                assignments[local] = Ineligible(Some(idx as u32));
            }
        }
        debug!("generator saved local assignments: {:?}", assignments);

        (ineligible_locals, assignments)
    }

    /// Compute the full generator layout.
    fn generator_layout(
        &self,
        ty: Ty<'tcx>,
        def_id: hir::def_id::DefId,
        substs: SubstsRef<'tcx>,
    ) -> Result<&'tcx Layout, LayoutError<'tcx>> {
        use SavedLocalEligibility::*;
        let tcx = self.tcx;
        let subst_field = |ty: Ty<'tcx>| ty.subst(tcx, substs);

        let info = match tcx.generator_layout(def_id) {
            None => return Err(LayoutError::Unknown(ty)),
            Some(info) => info,
        };
        let (ineligible_locals, assignments) = self.generator_saved_local_eligibility(&info);

        // Build a prefix layout, including "promoting" all ineligible
        // locals as part of the prefix. We compute the layout of all of
        // these fields at once to get optimal packing.
        let tag_index = substs.as_generator().prefix_tys().count();

        // `info.variant_fields` already accounts for the reserved variants, so no need to add them.
        let max_discr = (info.variant_fields.len() - 1) as u128;
        let discr_int = Integer::fit_unsigned(max_discr);
        let discr_int_ty = discr_int.to_ty(tcx, false);
        let tag = Scalar {
            value: Primitive::Int(discr_int, false),
            valid_range: WrappingRange { start: 0, end: max_discr },
        };
        let tag_layout = self.tcx.intern_layout(Layout::scalar(self, tag));
        let tag_layout = TyAndLayout { ty: discr_int_ty, layout: tag_layout };

        let promoted_layouts = ineligible_locals
            .iter()
            .map(|local| subst_field(info.field_tys[local]))
            .map(|ty| tcx.mk_maybe_uninit(ty))
            .map(|ty| self.layout_of(ty));
        let prefix_layouts = substs
            .as_generator()
            .prefix_tys()
            .map(|ty| self.layout_of(ty))
            .chain(iter::once(Ok(tag_layout)))
            .chain(promoted_layouts)
            .collect::<Result<Vec<_>, _>>()?;
        let prefix = self.univariant_uninterned(
            ty,
            &prefix_layouts,
            &ReprOptions::default(),
            StructKind::AlwaysSized,
        )?;

        let (prefix_size, prefix_align) = (prefix.size, prefix.align);

        // Split the prefix layout into the "outer" fields (upvars and
        // discriminant) and the "promoted" fields. Promoted fields will
        // get included in each variant that requested them in
        // GeneratorLayout.
        debug!("prefix = {:#?}", prefix);
        let (outer_fields, promoted_offsets, promoted_memory_index) = match prefix.fields {
            FieldsShape::Arbitrary { mut offsets, memory_index } => {
                let mut inverse_memory_index = invert_mapping(&memory_index);

                // "a" (`0..b_start`) and "b" (`b_start..`) correspond to
                // "outer" and "promoted" fields respectively.
                let b_start = (tag_index + 1) as u32;
                let offsets_b = offsets.split_off(b_start as usize);
                let offsets_a = offsets;

                // Disentangle the "a" and "b" components of `inverse_memory_index`
                // by preserving the order but keeping only one disjoint "half" each.
                // FIXME(eddyb) build a better abstraction for permutations, if possible.
                let inverse_memory_index_b: Vec<_> =
                    inverse_memory_index.iter().filter_map(|&i| i.checked_sub(b_start)).collect();
                inverse_memory_index.retain(|&i| i < b_start);
                let inverse_memory_index_a = inverse_memory_index;

                // Since `inverse_memory_index_{a,b}` each only refer to their
                // respective fields, they can be safely inverted
                let memory_index_a = invert_mapping(&inverse_memory_index_a);
                let memory_index_b = invert_mapping(&inverse_memory_index_b);

                let outer_fields =
                    FieldsShape::Arbitrary { offsets: offsets_a, memory_index: memory_index_a };
                (outer_fields, offsets_b, memory_index_b)
            }
            _ => bug!(),
        };

        let mut size = prefix.size;
        let mut align = prefix.align;
        let variants = info
            .variant_fields
            .iter_enumerated()
            .map(|(index, variant_fields)| {
                // Only include overlap-eligible fields when we compute our variant layout.
                let variant_only_tys = variant_fields
                    .iter()
                    .filter(|local| match assignments[**local] {
                        Unassigned => bug!(),
                        Assigned(v) if v == index => true,
                        Assigned(_) => bug!("assignment does not match variant"),
                        Ineligible(_) => false,
                    })
                    .map(|local| subst_field(info.field_tys[*local]));

                let mut variant = self.univariant_uninterned(
                    ty,
                    &variant_only_tys
                        .map(|ty| self.layout_of(ty))
                        .collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    StructKind::Prefixed(prefix_size, prefix_align.abi),
                )?;
                variant.variants = Variants::Single { index };

                let (offsets, memory_index) = match variant.fields {
                    FieldsShape::Arbitrary { offsets, memory_index } => (offsets, memory_index),
                    _ => bug!(),
                };

                // Now, stitch the promoted and variant-only fields back together in
                // the order they are mentioned by our GeneratorLayout.
                // Because we only use some subset (that can differ between variants)
                // of the promoted fields, we can't just pick those elements of the
                // `promoted_memory_index` (as we'd end up with gaps).
                // So instead, we build an "inverse memory_index", as if all of the
                // promoted fields were being used, but leave the elements not in the
                // subset as `INVALID_FIELD_IDX`, which we can filter out later to
                // obtain a valid (bijective) mapping.
                const INVALID_FIELD_IDX: u32 = !0;
                let mut combined_inverse_memory_index =
                    vec![INVALID_FIELD_IDX; promoted_memory_index.len() + memory_index.len()];
                let mut offsets_and_memory_index = iter::zip(offsets, memory_index);
                let combined_offsets = variant_fields
                    .iter()
                    .enumerate()
                    .map(|(i, local)| {
                        let (offset, memory_index) = match assignments[*local] {
                            Unassigned => bug!(),
                            Assigned(_) => {
                                let (offset, memory_index) =
                                    offsets_and_memory_index.next().unwrap();
                                (offset, promoted_memory_index.len() as u32 + memory_index)
                            }
                            Ineligible(field_idx) => {
                                let field_idx = field_idx.unwrap() as usize;
                                (promoted_offsets[field_idx], promoted_memory_index[field_idx])
                            }
                        };
                        combined_inverse_memory_index[memory_index as usize] = i as u32;
                        offset
                    })
                    .collect();

                // Remove the unused slots and invert the mapping to obtain the
                // combined `memory_index` (also see previous comment).
                combined_inverse_memory_index.retain(|&i| i != INVALID_FIELD_IDX);
                let combined_memory_index = invert_mapping(&combined_inverse_memory_index);

                variant.fields = FieldsShape::Arbitrary {
                    offsets: combined_offsets,
                    memory_index: combined_memory_index,
                };

                size = size.max(variant.size);
                align = align.max(variant.align);
                Ok(variant)
            })
            .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

        size = size.align_to(align.abi);

        let abi = if prefix.abi.is_uninhabited() || variants.iter().all(|v| v.abi.is_uninhabited())
        {
            Abi::Uninhabited
        } else {
            Abi::Aggregate { sized: true }
        };

        let layout = tcx.intern_layout(Layout {
            variants: Variants::Multiple {
                tag,
                tag_encoding: TagEncoding::Direct,
                tag_field: tag_index,
                variants,
            },
            fields: outer_fields,
            abi,
            largest_niche: prefix.largest_niche,
            size,
            align,
        });
        debug!("generator layout ({:?}): {:#?}", ty, layout);
        Ok(layout)
    }

    /// This is invoked by the `layout_of` query to record the final
    /// layout of each type.
    #[inline(always)]
    fn record_layout_for_printing(&self, layout: TyAndLayout<'tcx>) {
        // If we are running with `-Zprint-type-sizes`, maybe record layouts
        // for dumping later.
        if self.tcx.sess.opts.debugging_opts.print_type_sizes {
            self.record_layout_for_printing_outlined(layout)
        }
    }

    fn record_layout_for_printing_outlined(&self, layout: TyAndLayout<'tcx>) {
        // Ignore layouts that are done with non-empty environments or
        // non-monomorphic layouts, as the user only wants to see the stuff
        // resulting from the final codegen session.
        if layout.ty.definitely_has_param_types_or_consts(self.tcx)
            || !self.param_env.caller_bounds().is_empty()
        {
            return;
        }

        // (delay format until we actually need it)
        let record = |kind, packed, opt_discr_size, variants| {
            let type_desc = format!("{:?}", layout.ty);
            self.tcx.sess.code_stats.record_type_size(
                kind,
                type_desc,
                layout.align.abi,
                layout.size,
                packed,
                opt_discr_size,
                variants,
            );
        };

        let adt_def = match *layout.ty.kind() {
            ty::Adt(ref adt_def, _) => {
                debug!("print-type-size t: `{:?}` process adt", layout.ty);
                adt_def
            }

            ty::Closure(..) => {
                debug!("print-type-size t: `{:?}` record closure", layout.ty);
                record(DataTypeKind::Closure, false, None, vec![]);
                return;
            }

            _ => {
                debug!("print-type-size t: `{:?}` skip non-nominal", layout.ty);
                return;
            }
        };

        let adt_kind = adt_def.adt_kind();
        let adt_packed = adt_def.repr.pack.is_some();

        let build_variant_info = |n: Option<Ident>, flds: &[Symbol], layout: TyAndLayout<'tcx>| {
            let mut min_size = Size::ZERO;
            let field_info: Vec<_> = flds
                .iter()
                .enumerate()
                .map(|(i, &name)| {
                    let field_layout = layout.field(self, i);
                    let offset = layout.fields.offset(i);
                    let field_end = offset + field_layout.size;
                    if min_size < field_end {
                        min_size = field_end;
                    }
                    FieldInfo {
                        name: name.to_string(),
                        offset: offset.bytes(),
                        size: field_layout.size.bytes(),
                        align: field_layout.align.abi.bytes(),
                    }
                })
                .collect();

            VariantInfo {
                name: n.map(|n| n.to_string()),
                kind: if layout.is_unsized() { SizeKind::Min } else { SizeKind::Exact },
                align: layout.align.abi.bytes(),
                size: if min_size.bytes() == 0 { layout.size.bytes() } else { min_size.bytes() },
                fields: field_info,
            }
        };

        match layout.variants {
            Variants::Single { index } => {
                if !adt_def.variants.is_empty() && layout.fields != FieldsShape::Primitive {
                    debug!(
                        "print-type-size `{:#?}` variant {}",
                        layout, adt_def.variants[index].ident
                    );
                    let variant_def = &adt_def.variants[index];
                    let fields: Vec<_> = variant_def.fields.iter().map(|f| f.ident.name).collect();
                    record(
                        adt_kind.into(),
                        adt_packed,
                        None,
                        vec![build_variant_info(Some(variant_def.ident), &fields, layout)],
                    );
                } else {
                    // (This case arises for *empty* enums; so give it
                    // zero variants.)
                    record(adt_kind.into(), adt_packed, None, vec![]);
                }
            }

            Variants::Multiple { tag, ref tag_encoding, .. } => {
                debug!(
                    "print-type-size `{:#?}` adt general variants def {}",
                    layout.ty,
                    adt_def.variants.len()
                );
                let variant_infos: Vec<_> = adt_def
                    .variants
                    .iter_enumerated()
                    .map(|(i, variant_def)| {
                        let fields: Vec<_> =
                            variant_def.fields.iter().map(|f| f.ident.name).collect();
                        build_variant_info(
                            Some(variant_def.ident),
                            &fields,
                            layout.for_variant(self, i),
                        )
                    })
                    .collect();
                record(
                    adt_kind.into(),
                    adt_packed,
                    match tag_encoding {
                        TagEncoding::Direct => Some(tag.value.size(self)),
                        _ => None,
                    },
                    variant_infos,
                );
            }
        }
    }
}

/// Type size "skeleton", i.e., the only information determining a type's size.
/// While this is conservative, (aside from constant sizes, only pointers,
/// newtypes thereof and null pointer optimized enums are allowed), it is
/// enough to statically check common use cases of transmute.
#[derive(Copy, Clone, Debug)]
pub enum SizeSkeleton<'tcx> {
    /// Any statically computable Layout.
    Known(Size),

    /// A potentially-fat pointer.
    Pointer {
        /// If true, this pointer is never null.
        non_zero: bool,
        /// The type which determines the unsized metadata, if any,
        /// of this pointer. Either a type parameter or a projection
        /// depending on one, with regions erased.
        tail: Ty<'tcx>,
    },
}

impl<'tcx> SizeSkeleton<'tcx> {
    pub fn compute(
        ty: Ty<'tcx>,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Result<SizeSkeleton<'tcx>, LayoutError<'tcx>> {
        debug_assert!(!ty.has_infer_types_or_consts());

        // First try computing a static layout.
        let err = match tcx.layout_of(param_env.and(ty)) {
            Ok(layout) => {
                return Ok(SizeSkeleton::Known(layout.size));
            }
            Err(err) => err,
        };

        match *ty.kind() {
            ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                let non_zero = !ty.is_unsafe_ptr();
                let tail = tcx.struct_tail_erasing_lifetimes(pointee, param_env);
                match tail.kind() {
                    ty::Param(_) | ty::Projection(_) => {
                        debug_assert!(tail.definitely_has_param_types_or_consts(tcx));
                        Ok(SizeSkeleton::Pointer { non_zero, tail: tcx.erase_regions(tail) })
                    }
                    _ => bug!(
                        "SizeSkeleton::compute({}): layout errored ({}), yet \
                              tail `{}` is not a type parameter or a projection",
                        ty,
                        err,
                        tail
                    ),
                }
            }

            ty::Adt(def, substs) => {
                // Only newtypes and enums w/ nullable pointer optimization.
                if def.is_union() || def.variants.is_empty() || def.variants.len() > 2 {
                    return Err(err);
                }

                // Get a zero-sized variant or a pointer newtype.
                let zero_or_ptr_variant = |i| {
                    let i = VariantIdx::new(i);
                    let fields = def.variants[i]
                        .fields
                        .iter()
                        .map(|field| SizeSkeleton::compute(field.ty(tcx, substs), tcx, param_env));
                    let mut ptr = None;
                    for field in fields {
                        let field = field?;
                        match field {
                            SizeSkeleton::Known(size) => {
                                if size.bytes() > 0 {
                                    return Err(err);
                                }
                            }
                            SizeSkeleton::Pointer { .. } => {
                                if ptr.is_some() {
                                    return Err(err);
                                }
                                ptr = Some(field);
                            }
                        }
                    }
                    Ok(ptr)
                };

                let v0 = zero_or_ptr_variant(0)?;
                // Newtype.
                if def.variants.len() == 1 {
                    if let Some(SizeSkeleton::Pointer { non_zero, tail }) = v0 {
                        return Ok(SizeSkeleton::Pointer {
                            non_zero: non_zero
                                || match tcx.layout_scalar_valid_range(def.did) {
                                    (Bound::Included(start), Bound::Unbounded) => start > 0,
                                    (Bound::Included(start), Bound::Included(end)) => {
                                        0 < start && start < end
                                    }
                                    _ => false,
                                },
                            tail,
                        });
                    } else {
                        return Err(err);
                    }
                }

                let v1 = zero_or_ptr_variant(1)?;
                // Nullable pointer enum optimization.
                match (v0, v1) {
                    (Some(SizeSkeleton::Pointer { non_zero: true, tail }), None)
                    | (None, Some(SizeSkeleton::Pointer { non_zero: true, tail })) => {
                        Ok(SizeSkeleton::Pointer { non_zero: false, tail })
                    }
                    _ => Err(err),
                }
            }

            ty::Projection(_) | ty::Opaque(..) => {
                let normalized = tcx.normalize_erasing_regions(param_env, ty);
                if ty == normalized {
                    Err(err)
                } else {
                    SizeSkeleton::compute(normalized, tcx, param_env)
                }
            }

            _ => Err(err),
        }
    }

    pub fn same_size(self, other: SizeSkeleton<'_>) -> bool {
        match (self, other) {
            (SizeSkeleton::Known(a), SizeSkeleton::Known(b)) => a == b,
            (SizeSkeleton::Pointer { tail: a, .. }, SizeSkeleton::Pointer { tail: b, .. }) => {
                a == b
            }
            _ => false,
        }
    }
}

pub trait HasTyCtxt<'tcx>: HasDataLayout {
    fn tcx(&self) -> TyCtxt<'tcx>;
}

pub trait HasParamEnv<'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx>;
}

impl<'tcx> HasDataLayout for TyCtxt<'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.data_layout
    }
}

impl<'tcx> HasTargetSpec for TyCtxt<'tcx> {
    fn target_spec(&self) -> &Target {
        &self.sess.target
    }
}

impl<'tcx> HasTyCtxt<'tcx> for TyCtxt<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        *self
    }
}

impl<'tcx> HasDataLayout for ty::query::TyCtxtAt<'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.data_layout
    }
}

impl<'tcx> HasTargetSpec for ty::query::TyCtxtAt<'tcx> {
    fn target_spec(&self) -> &Target {
        &self.sess.target
    }
}

impl<'tcx> HasTyCtxt<'tcx> for ty::query::TyCtxtAt<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        **self
    }
}

impl<'tcx, C> HasParamEnv<'tcx> for LayoutCx<'tcx, C> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'tcx, T: HasDataLayout> HasDataLayout for LayoutCx<'tcx, T> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.tcx.data_layout()
    }
}

impl<'tcx, T: HasTargetSpec> HasTargetSpec for LayoutCx<'tcx, T> {
    fn target_spec(&self) -> &Target {
        self.tcx.target_spec()
    }
}

impl<'tcx, T: HasTyCtxt<'tcx>> HasTyCtxt<'tcx> for LayoutCx<'tcx, T> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx.tcx()
    }
}

pub trait MaybeResult<T> {
    type Error;

    fn from(x: Result<T, Self::Error>) -> Self;
    fn to_result(self) -> Result<T, Self::Error>;
}

impl<T> MaybeResult<T> for T {
    type Error = !;

    fn from(Ok(x): Result<T, Self::Error>) -> Self {
        x
    }
    fn to_result(self) -> Result<T, Self::Error> {
        Ok(self)
    }
}

impl<T, E> MaybeResult<T> for Result<T, E> {
    type Error = E;

    fn from(x: Result<T, Self::Error>) -> Self {
        x
    }
    fn to_result(self) -> Result<T, Self::Error> {
        self
    }
}

pub type TyAndLayout<'tcx> = rustc_target::abi::TyAndLayout<'tcx, Ty<'tcx>>;

/// Trait for contexts that want to be able to compute layouts of types.
/// This automatically gives access to `LayoutOf`, through a blanket `impl`.
pub trait LayoutOfHelpers<'tcx>: HasDataLayout + HasTyCtxt<'tcx> + HasParamEnv<'tcx> {
    /// The `TyAndLayout`-wrapping type (or `TyAndLayout` itself), which will be
    /// returned from `layout_of` (see also `handle_layout_err`).
    type LayoutOfResult: MaybeResult<TyAndLayout<'tcx>>;

    /// `Span` to use for `tcx.at(span)`, from `layout_of`.
    // FIXME(eddyb) perhaps make this mandatory to get contexts to track it better?
    #[inline]
    fn layout_tcx_at_span(&self) -> Span {
        DUMMY_SP
    }

    /// Helper used for `layout_of`, to adapt `tcx.layout_of(...)` into a
    /// `Self::LayoutOfResult` (which does not need to be a `Result<...>`).
    ///
    /// Most `impl`s, which propagate `LayoutError`s, should simply return `err`,
    /// but this hook allows e.g. codegen to return only `TyAndLayout` from its
    /// `cx.layout_of(...)`, without any `Result<...>` around it to deal with
    /// (and any `LayoutError`s are turned into fatal errors or ICEs).
    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        span: Span,
        ty: Ty<'tcx>,
    ) -> <Self::LayoutOfResult as MaybeResult<TyAndLayout<'tcx>>>::Error;
}

/// Blanket extension trait for contexts that can compute layouts of types.
pub trait LayoutOf<'tcx>: LayoutOfHelpers<'tcx> {
    /// Computes the layout of a type. Note that this implicitly
    /// executes in "reveal all" mode, and will normalize the input type.
    #[inline]
    fn layout_of(&self, ty: Ty<'tcx>) -> Self::LayoutOfResult {
        self.spanned_layout_of(ty, DUMMY_SP)
    }

    /// Computes the layout of a type, at `span`. Note that this implicitly
    /// executes in "reveal all" mode, and will normalize the input type.
    // FIXME(eddyb) avoid passing information like this, and instead add more
    // `TyCtxt::at`-like APIs to be able to do e.g. `cx.at(span).layout_of(ty)`.
    #[inline]
    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::LayoutOfResult {
        let span = if !span.is_dummy() { span } else { self.layout_tcx_at_span() };
        let tcx = self.tcx().at(span);

        MaybeResult::from(
            tcx.layout_of(self.param_env().and(ty))
                .map_err(|err| self.handle_layout_err(err, span, ty)),
        )
    }
}

impl<C: LayoutOfHelpers<'tcx>> LayoutOf<'tcx> for C {}

impl LayoutOfHelpers<'tcx> for LayoutCx<'tcx, TyCtxt<'tcx>> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, _: Span, _: Ty<'tcx>) -> LayoutError<'tcx> {
        err
    }
}

impl LayoutOfHelpers<'tcx> for LayoutCx<'tcx, ty::query::TyCtxtAt<'tcx>> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    #[inline]
    fn layout_tcx_at_span(&self) -> Span {
        self.tcx.span
    }

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, _: Span, _: Ty<'tcx>) -> LayoutError<'tcx> {
        err
    }
}

impl<'tcx, C> TyAbiInterface<'tcx, C> for Ty<'tcx>
where
    C: HasTyCtxt<'tcx> + HasParamEnv<'tcx>,
{
    fn ty_and_layout_for_variant(
        this: TyAndLayout<'tcx>,
        cx: &C,
        variant_index: VariantIdx,
    ) -> TyAndLayout<'tcx> {
        let layout = match this.variants {
            Variants::Single { index }
                // If all variants but one are uninhabited, the variant layout is the enum layout.
                if index == variant_index &&
                // Don't confuse variants of uninhabited enums with the enum itself.
                // For more details see https://github.com/rust-lang/rust/issues/69763.
                this.fields != FieldsShape::Primitive =>
            {
                this.layout
            }

            Variants::Single { index } => {
                let tcx = cx.tcx();
                let param_env = cx.param_env();

                // Deny calling for_variant more than once for non-Single enums.
                if let Ok(original_layout) = tcx.layout_of(param_env.and(this.ty)) {
                    assert_eq!(original_layout.variants, Variants::Single { index });
                }

                let fields = match this.ty.kind() {
                    ty::Adt(def, _) if def.variants.is_empty() =>
                        bug!("for_variant called on zero-variant enum"),
                    ty::Adt(def, _) => def.variants[variant_index].fields.len(),
                    _ => bug!(),
                };
                tcx.intern_layout(Layout {
                    variants: Variants::Single { index: variant_index },
                    fields: match NonZeroUsize::new(fields) {
                        Some(fields) => FieldsShape::Union(fields),
                        None => FieldsShape::Arbitrary { offsets: vec![], memory_index: vec![] },
                    },
                    abi: Abi::Uninhabited,
                    largest_niche: None,
                    align: tcx.data_layout.i8_align,
                    size: Size::ZERO,
                })
            }

            Variants::Multiple { ref variants, .. } => &variants[variant_index],
        };

        assert_eq!(layout.variants, Variants::Single { index: variant_index });

        TyAndLayout { ty: this.ty, layout }
    }

    fn ty_and_layout_field(this: TyAndLayout<'tcx>, cx: &C, i: usize) -> TyAndLayout<'tcx> {
        enum TyMaybeWithLayout<'tcx> {
            Ty(Ty<'tcx>),
            TyAndLayout(TyAndLayout<'tcx>),
        }

        fn field_ty_or_layout(
            this: TyAndLayout<'tcx>,
            cx: &(impl HasTyCtxt<'tcx> + HasParamEnv<'tcx>),
            i: usize,
        ) -> TyMaybeWithLayout<'tcx> {
            let tcx = cx.tcx();
            let tag_layout = |tag: Scalar| -> TyAndLayout<'tcx> {
                let layout = Layout::scalar(cx, tag);
                TyAndLayout { layout: tcx.intern_layout(layout), ty: tag.value.to_ty(tcx) }
            };

            match *this.ty.kind() {
                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::FnPtr(_)
                | ty::Never
                | ty::FnDef(..)
                | ty::GeneratorWitness(..)
                | ty::Foreign(..)
                | ty::Dynamic(..) => bug!("TyAndLayout::field({:?}): not applicable", this),

                // Potentially-fat pointers.
                ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                    assert!(i < this.fields.count());

                    // Reuse the fat `*T` type as its own thin pointer data field.
                    // This provides information about, e.g., DST struct pointees
                    // (which may have no non-DST form), and will work as long
                    // as the `Abi` or `FieldsShape` is checked by users.
                    if i == 0 {
                        let nil = tcx.mk_unit();
                        let unit_ptr_ty = if this.ty.is_unsafe_ptr() {
                            tcx.mk_mut_ptr(nil)
                        } else {
                            tcx.mk_mut_ref(tcx.lifetimes.re_static, nil)
                        };

                        // NOTE(eddyb) using an empty `ParamEnv`, and `unwrap`-ing
                        // the `Result` should always work because the type is
                        // always either `*mut ()` or `&'static mut ()`.
                        return TyMaybeWithLayout::TyAndLayout(TyAndLayout {
                            ty: this.ty,
                            ..tcx.layout_of(ty::ParamEnv::reveal_all().and(unit_ptr_ty)).unwrap()
                        });
                    }

                    match tcx.struct_tail_erasing_lifetimes(pointee, cx.param_env()).kind() {
                        ty::Slice(_) | ty::Str => TyMaybeWithLayout::Ty(tcx.types.usize),
                        ty::Dynamic(_, _) => {
                            TyMaybeWithLayout::Ty(tcx.mk_imm_ref(
                                tcx.lifetimes.re_static,
                                tcx.mk_array(tcx.types.usize, 3),
                            ))
                            /* FIXME: use actual fn pointers
                            Warning: naively computing the number of entries in the
                            vtable by counting the methods on the trait + methods on
                            all parent traits does not work, because some methods can
                            be not object safe and thus excluded from the vtable.
                            Increase this counter if you tried to implement this but
                            failed to do it without duplicating a lot of code from
                            other places in the compiler: 2
                            tcx.mk_tup(&[
                                tcx.mk_array(tcx.types.usize, 3),
                                tcx.mk_array(Option<fn()>),
                            ])
                            */
                        }
                        _ => bug!("TyAndLayout::field({:?}): not applicable", this),
                    }
                }

                // Arrays and slices.
                ty::Array(element, _) | ty::Slice(element) => TyMaybeWithLayout::Ty(element),
                ty::Str => TyMaybeWithLayout::Ty(tcx.types.u8),

                // Tuples, generators and closures.
                ty::Closure(_, ref substs) => field_ty_or_layout(
                    TyAndLayout { ty: substs.as_closure().tupled_upvars_ty(), ..this },
                    cx,
                    i,
                ),

                ty::Generator(def_id, ref substs, _) => match this.variants {
                    Variants::Single { index } => TyMaybeWithLayout::Ty(
                        substs
                            .as_generator()
                            .state_tys(def_id, tcx)
                            .nth(index.as_usize())
                            .unwrap()
                            .nth(i)
                            .unwrap(),
                    ),
                    Variants::Multiple { tag, tag_field, .. } => {
                        if i == tag_field {
                            return TyMaybeWithLayout::TyAndLayout(tag_layout(tag));
                        }
                        TyMaybeWithLayout::Ty(substs.as_generator().prefix_tys().nth(i).unwrap())
                    }
                },

                ty::Tuple(tys) => TyMaybeWithLayout::Ty(tys[i].expect_ty()),

                // ADTs.
                ty::Adt(def, substs) => {
                    match this.variants {
                        Variants::Single { index } => {
                            TyMaybeWithLayout::Ty(def.variants[index].fields[i].ty(tcx, substs))
                        }

                        // Discriminant field for enums (where applicable).
                        Variants::Multiple { tag, .. } => {
                            assert_eq!(i, 0);
                            return TyMaybeWithLayout::TyAndLayout(tag_layout(tag));
                        }
                    }
                }

                ty::Projection(_)
                | ty::Bound(..)
                | ty::Placeholder(..)
                | ty::Opaque(..)
                | ty::Param(_)
                | ty::Infer(_)
                | ty::Error(_) => bug!("TyAndLayout::field: unexpected type `{}`", this.ty),
            }
        }

        match field_ty_or_layout(this, cx, i) {
            TyMaybeWithLayout::Ty(field_ty) => {
                cx.tcx().layout_of(cx.param_env().and(field_ty)).unwrap_or_else(|e| {
                    bug!(
                        "failed to get layout for `{}`: {},\n\
                         despite it being a field (#{}) of an existing layout: {:#?}",
                        field_ty,
                        e,
                        i,
                        this
                    )
                })
            }
            TyMaybeWithLayout::TyAndLayout(field_layout) => field_layout,
        }
    }

    fn ty_and_layout_pointee_info_at(
        this: TyAndLayout<'tcx>,
        cx: &C,
        offset: Size,
    ) -> Option<PointeeInfo> {
        let tcx = cx.tcx();
        let param_env = cx.param_env();

        let addr_space_of_ty = |ty: Ty<'tcx>| {
            if ty.is_fn() { cx.data_layout().instruction_address_space } else { AddressSpace::DATA }
        };

        let pointee_info = match *this.ty.kind() {
            ty::RawPtr(mt) if offset.bytes() == 0 => {
                tcx.layout_of(param_env.and(mt.ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: None,
                    address_space: addr_space_of_ty(mt.ty),
                })
            }
            ty::FnPtr(fn_sig) if offset.bytes() == 0 => {
                tcx.layout_of(param_env.and(tcx.mk_fn_ptr(fn_sig))).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: None,
                    address_space: cx.data_layout().instruction_address_space,
                })
            }
            ty::Ref(_, ty, mt) if offset.bytes() == 0 => {
                let address_space = addr_space_of_ty(ty);
                let kind = if tcx.sess.opts.optimize == OptLevel::No {
                    // Use conservative pointer kind if not optimizing. This saves us the
                    // Freeze/Unpin queries, and can save time in the codegen backend (noalias
                    // attributes in LLVM have compile-time cost even in unoptimized builds).
                    PointerKind::Shared
                } else {
                    match mt {
                        hir::Mutability::Not => {
                            if ty.is_freeze(tcx.at(DUMMY_SP), cx.param_env()) {
                                PointerKind::Frozen
                            } else {
                                PointerKind::Shared
                            }
                        }
                        hir::Mutability::Mut => {
                            // References to self-referential structures should not be considered
                            // noalias, as another pointer to the structure can be obtained, that
                            // is not based-on the original reference. We consider all !Unpin
                            // types to be potentially self-referential here.
                            if ty.is_unpin(tcx.at(DUMMY_SP), cx.param_env()) {
                                PointerKind::UniqueBorrowed
                            } else {
                                PointerKind::Shared
                            }
                        }
                    }
                };

                tcx.layout_of(param_env.and(ty)).ok().map(|layout| PointeeInfo {
                    size: layout.size,
                    align: layout.align.abi,
                    safe: Some(kind),
                    address_space,
                })
            }

            _ => {
                let mut data_variant = match this.variants {
                    // Within the discriminant field, only the niche itself is
                    // always initialized, so we only check for a pointer at its
                    // offset.
                    //
                    // If the niche is a pointer, it's either valid (according
                    // to its type), or null (which the niche field's scalar
                    // validity range encodes).  This allows using
                    // `dereferenceable_or_null` for e.g., `Option<&T>`, and
                    // this will continue to work as long as we don't start
                    // using more niches than just null (e.g., the first page of
                    // the address space, or unaligned pointers).
                    Variants::Multiple {
                        tag_encoding: TagEncoding::Niche { dataful_variant, .. },
                        tag_field,
                        ..
                    } if this.fields.offset(tag_field) == offset => {
                        Some(this.for_variant(cx, dataful_variant))
                    }
                    _ => Some(this),
                };

                if let Some(variant) = data_variant {
                    // We're not interested in any unions.
                    if let FieldsShape::Union(_) = variant.fields {
                        data_variant = None;
                    }
                }

                let mut result = None;

                if let Some(variant) = data_variant {
                    let ptr_end = offset + Pointer.size(cx);
                    for i in 0..variant.fields.count() {
                        let field_start = variant.fields.offset(i);
                        if field_start <= offset {
                            let field = variant.field(cx, i);
                            result = field.to_result().ok().and_then(|field| {
                                if ptr_end <= field_start + field.size {
                                    // We found the right field, look inside it.
                                    let field_info =
                                        field.pointee_info_at(cx, offset - field_start);
                                    field_info
                                } else {
                                    None
                                }
                            });
                            if result.is_some() {
                                break;
                            }
                        }
                    }
                }

                // FIXME(eddyb) This should be for `ptr::Unique<T>`, not `Box<T>`.
                if let Some(ref mut pointee) = result {
                    if let ty::Adt(def, _) = this.ty.kind() {
                        if def.is_box() && offset.bytes() == 0 {
                            pointee.safe = Some(PointerKind::UniqueOwned);
                        }
                    }
                }

                result
            }
        };

        debug!(
            "pointee_info_at (offset={:?}, type kind: {:?}) => {:?}",
            offset,
            this.ty.kind(),
            pointee_info
        );

        pointee_info
    }
}

impl<'tcx> ty::Instance<'tcx> {
    // NOTE(eddyb) this is private to avoid using it from outside of
    // `fn_abi_of_instance` - any other uses are either too high-level
    // for `Instance` (e.g. typeck would use `Ty::fn_sig` instead),
    // or should go through `FnAbi` instead, to avoid losing any
    // adjustments `fn_abi_of_instance` might be performing.
    fn fn_sig_for_fn_abi(&self, tcx: TyCtxt<'tcx>) -> ty::PolyFnSig<'tcx> {
        // FIXME(davidtwco,eddyb): A `ParamEnv` should be passed through to this function.
        let ty = self.ty(tcx, ty::ParamEnv::reveal_all());
        match *ty.kind() {
            ty::FnDef(..) => {
                // HACK(davidtwco,eddyb): This is a workaround for polymorphization considering
                // parameters unused if they show up in the signature, but not in the `mir::Body`
                // (i.e. due to being inside a projection that got normalized, see
                // `src/test/ui/polymorphization/normalized_sig_types.rs`), and codegen not keeping
                // track of a polymorphization `ParamEnv` to allow normalizing later.
                let mut sig = match *ty.kind() {
                    ty::FnDef(def_id, substs) => tcx
                        .normalize_erasing_regions(tcx.param_env(def_id), tcx.fn_sig(def_id))
                        .subst(tcx, substs),
                    _ => unreachable!(),
                };

                if let ty::InstanceDef::VtableShim(..) = self.def {
                    // Modify `fn(self, ...)` to `fn(self: *mut Self, ...)`.
                    sig = sig.map_bound(|mut sig| {
                        let mut inputs_and_output = sig.inputs_and_output.to_vec();
                        inputs_and_output[0] = tcx.mk_mut_ptr(inputs_and_output[0]);
                        sig.inputs_and_output = tcx.intern_type_list(&inputs_and_output);
                        sig
                    });
                }
                sig
            }
            ty::Closure(def_id, substs) => {
                let sig = substs.as_closure().sig();

                let bound_vars = tcx.mk_bound_variable_kinds(
                    sig.bound_vars()
                        .iter()
                        .chain(iter::once(ty::BoundVariableKind::Region(ty::BrEnv))),
                );
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                    kind: ty::BoundRegionKind::BrEnv,
                };
                let env_region = ty::ReLateBound(ty::INNERMOST, br);
                let env_ty = tcx.closure_env_ty(def_id, substs, env_region).unwrap();

                let sig = sig.skip_binder();
                ty::Binder::bind_with_vars(
                    tcx.mk_fn_sig(
                        iter::once(env_ty).chain(sig.inputs().iter().cloned()),
                        sig.output(),
                        sig.c_variadic,
                        sig.unsafety,
                        sig.abi,
                    ),
                    bound_vars,
                )
            }
            ty::Generator(_, substs, _) => {
                let sig = substs.as_generator().poly_sig();

                let bound_vars = tcx.mk_bound_variable_kinds(
                    sig.bound_vars()
                        .iter()
                        .chain(iter::once(ty::BoundVariableKind::Region(ty::BrEnv))),
                );
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                    kind: ty::BoundRegionKind::BrEnv,
                };
                let env_region = ty::ReLateBound(ty::INNERMOST, br);
                let env_ty = tcx.mk_mut_ref(tcx.mk_region(env_region), ty);

                let pin_did = tcx.require_lang_item(LangItem::Pin, None);
                let pin_adt_ref = tcx.adt_def(pin_did);
                let pin_substs = tcx.intern_substs(&[env_ty.into()]);
                let env_ty = tcx.mk_adt(pin_adt_ref, pin_substs);

                let sig = sig.skip_binder();
                let state_did = tcx.require_lang_item(LangItem::GeneratorState, None);
                let state_adt_ref = tcx.adt_def(state_did);
                let state_substs = tcx.intern_substs(&[sig.yield_ty.into(), sig.return_ty.into()]);
                let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);
                ty::Binder::bind_with_vars(
                    tcx.mk_fn_sig(
                        [env_ty, sig.resume_ty].iter(),
                        &ret_ty,
                        false,
                        hir::Unsafety::Normal,
                        rustc_target::spec::abi::Abi::Rust,
                    ),
                    bound_vars,
                )
            }
            _ => bug!("unexpected type {:?} in Instance::fn_sig", ty),
        }
    }
}

/// Calculates whether a function's ABI can unwind or not.
///
/// This takes two primary parameters:
///
/// * `codegen_fn_attr_flags` - these are flags calculated as part of the
///   codegen attrs for a defined function. For function pointers this set of
///   flags is the empty set. This is only applicable for Rust-defined
///   functions, and generally isn't needed except for small optimizations where
///   we try to say a function which otherwise might look like it could unwind
///   doesn't actually unwind (such as for intrinsics and such).
///
/// * `abi` - this is the ABI that the function is defined with. This is the
///   primary factor for determining whether a function can unwind or not.
///
/// Note that in this case unwinding is not necessarily panicking in Rust. Rust
/// panics are implemented with unwinds on most platform (when
/// `-Cpanic=unwind`), but this also accounts for `-Cpanic=abort` build modes.
/// Notably unwinding is disallowed for more non-Rust ABIs unless it's
/// specifically in the name (e.g. `"C-unwind"`). Unwinding within each ABI is
/// defined for each ABI individually, but it always corresponds to some form of
/// stack-based unwinding (the exact mechanism of which varies
/// platform-by-platform).
///
/// Rust functions are classfied whether or not they can unwind based on the
/// active "panic strategy". In other words Rust functions are considered to
/// unwind in `-Cpanic=unwind` mode and cannot unwind in `-Cpanic=abort` mode.
/// Note that Rust supports intermingling panic=abort and panic=unwind code, but
/// only if the final panic mode is panic=abort. In this scenario any code
/// previously compiled assuming that a function can unwind is still correct, it
/// just never happens to actually unwind at runtime.
///
/// This function's answer to whether or not a function can unwind is quite
/// impactful throughout the compiler. This affects things like:
///
/// * Calling a function which can't unwind means codegen simply ignores any
///   associated unwinding cleanup.
/// * Calling a function which can unwind from a function which can't unwind
///   causes the `abort_unwinding_calls` MIR pass to insert a landing pad that
///   aborts the process.
/// * This affects whether functions have the LLVM `nounwind` attribute, which
///   affects various optimizations and codegen.
///
/// FIXME: this is actually buggy with respect to Rust functions. Rust functions
/// compiled with `-Cpanic=unwind` and referenced from another crate compiled
/// with `-Cpanic=abort` will look like they can't unwind when in fact they
/// might (from a foreign exception or similar).
#[inline]
pub fn fn_can_unwind(
    tcx: TyCtxt<'tcx>,
    codegen_fn_attr_flags: CodegenFnAttrFlags,
    abi: SpecAbi,
) -> bool {
    // Special attribute for functions which can't unwind.
    if codegen_fn_attr_flags.contains(CodegenFnAttrFlags::NEVER_UNWIND) {
        return false;
    }

    // Otherwise if this isn't special then unwinding is generally determined by
    // the ABI of the itself. ABIs like `C` have variants which also
    // specifically allow unwinding (`C-unwind`), but not all platform-specific
    // ABIs have such an option. Otherwise the only other thing here is Rust
    // itself, and those ABIs are determined by the panic strategy configured
    // for this compilation.
    //
    // Unfortunately at this time there's also another caveat. Rust [RFC
    // 2945][rfc] has been accepted and is in the process of being implemented
    // and stabilized. In this interim state we need to deal with historical
    // rustc behavior as well as plan for future rustc behavior.
    //
    // Historically functions declared with `extern "C"` were marked at the
    // codegen layer as `nounwind`. This happened regardless of `panic=unwind`
    // or not. This is UB for functions in `panic=unwind` mode that then
    // actually panic and unwind. Note that this behavior is true for both
    // externally declared functions as well as Rust-defined function.
    //
    // To fix this UB rustc would like to change in the future to catch unwinds
    // from function calls that may unwind within a Rust-defined `extern "C"`
    // function and forcibly abort the process, thereby respecting the
    // `nounwind` attribut emitted for `extern "C"`. This behavior change isn't
    // ready to roll out, so determining whether or not the `C` family of ABIs
    // unwinds is conditional not only on their definition but also whether the
    // `#![feature(c_unwind)]` feature gate is active.
    //
    // Note that this means that unlike historical compilers rustc now, by
    // default, unconditionally thinks that the `C` ABI may unwind. This will
    // prevent some optimization opportunities, however, so we try to scope this
    // change and only assume that `C` unwinds with `panic=unwind` (as opposed
    // to `panic=abort`).
    //
    // Eventually the check against `c_unwind` here will ideally get removed and
    // this'll be a little cleaner as it'll be a straightforward check of the
    // ABI.
    //
    // [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/2945-c-unwind-abi.md
    use SpecAbi::*;
    match abi {
        C { unwind } | Stdcall { unwind } | System { unwind } | Thiscall { unwind } => {
            unwind
                || (!tcx.features().c_unwind && tcx.sess.panic_strategy() == PanicStrategy::Unwind)
        }
        Cdecl
        | Fastcall
        | Vectorcall
        | Aapcs
        | Win64
        | SysV64
        | PtxKernel
        | Msp430Interrupt
        | X86Interrupt
        | AmdGpuKernel
        | EfiApi
        | AvrInterrupt
        | AvrNonBlockingInterrupt
        | CCmseNonSecureCall
        | Wasm
        | RustIntrinsic
        | PlatformIntrinsic
        | Unadjusted => false,
        Rust | RustCall => tcx.sess.panic_strategy() == PanicStrategy::Unwind,
    }
}

#[inline]
pub fn conv_from_spec_abi(tcx: TyCtxt<'_>, abi: SpecAbi) -> Conv {
    use rustc_target::spec::abi::Abi::*;
    match tcx.sess.target.adjust_abi(abi) {
        RustIntrinsic | PlatformIntrinsic | Rust | RustCall => Conv::Rust,

        // It's the ABI's job to select this, not ours.
        System { .. } => bug!("system abi should be selected elsewhere"),
        EfiApi => bug!("eficall abi should be selected elsewhere"),

        Stdcall { .. } => Conv::X86Stdcall,
        Fastcall => Conv::X86Fastcall,
        Vectorcall => Conv::X86VectorCall,
        Thiscall { .. } => Conv::X86ThisCall,
        C { .. } => Conv::C,
        Unadjusted => Conv::C,
        Win64 => Conv::X86_64Win64,
        SysV64 => Conv::X86_64SysV,
        Aapcs => Conv::ArmAapcs,
        CCmseNonSecureCall => Conv::CCmseNonSecureCall,
        PtxKernel => Conv::PtxKernel,
        Msp430Interrupt => Conv::Msp430Intr,
        X86Interrupt => Conv::X86Intr,
        AmdGpuKernel => Conv::AmdGpuKernel,
        AvrInterrupt => Conv::AvrInterrupt,
        AvrNonBlockingInterrupt => Conv::AvrNonBlockingInterrupt,
        Wasm => Conv::C,

        // These API constants ought to be more specific...
        Cdecl => Conv::C,
    }
}

/// Error produced by attempting to compute or adjust a `FnAbi`.
#[derive(Clone, Debug, HashStable)]
pub enum FnAbiError<'tcx> {
    /// Error produced by a `layout_of` call, while computing `FnAbi` initially.
    Layout(LayoutError<'tcx>),

    /// Error produced by attempting to adjust a `FnAbi`, for a "foreign" ABI.
    AdjustForForeignAbi(call::AdjustForForeignAbiError),
}

impl From<LayoutError<'tcx>> for FnAbiError<'tcx> {
    fn from(err: LayoutError<'tcx>) -> Self {
        Self::Layout(err)
    }
}

impl From<call::AdjustForForeignAbiError> for FnAbiError<'_> {
    fn from(err: call::AdjustForForeignAbiError) -> Self {
        Self::AdjustForForeignAbi(err)
    }
}

impl<'tcx> fmt::Display for FnAbiError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(err) => err.fmt(f),
            Self::AdjustForForeignAbi(err) => err.fmt(f),
        }
    }
}

// FIXME(eddyb) maybe use something like this for an unified `fn_abi_of`, not
// just for error handling.
#[derive(Debug)]
pub enum FnAbiRequest<'tcx> {
    OfFnPtr { sig: ty::PolyFnSig<'tcx>, extra_args: &'tcx ty::List<Ty<'tcx>> },
    OfInstance { instance: ty::Instance<'tcx>, extra_args: &'tcx ty::List<Ty<'tcx>> },
}

/// Trait for contexts that want to be able to compute `FnAbi`s.
/// This automatically gives access to `FnAbiOf`, through a blanket `impl`.
pub trait FnAbiOfHelpers<'tcx>: LayoutOfHelpers<'tcx> {
    /// The `&FnAbi`-wrapping type (or `&FnAbi` itself), which will be
    /// returned from `fn_abi_of_*` (see also `handle_fn_abi_err`).
    type FnAbiOfResult: MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>>;

    /// Helper used for `fn_abi_of_*`, to adapt `tcx.fn_abi_of_*(...)` into a
    /// `Self::FnAbiOfResult` (which does not need to be a `Result<...>`).
    ///
    /// Most `impl`s, which propagate `FnAbiError`s, should simply return `err`,
    /// but this hook allows e.g. codegen to return only `&FnAbi` from its
    /// `cx.fn_abi_of_*(...)`, without any `Result<...>` around it to deal with
    /// (and any `FnAbiError`s are turned into fatal errors or ICEs).
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>>>::Error;
}

/// Blanket extension trait for contexts that can compute `FnAbi`s.
pub trait FnAbiOf<'tcx>: FnAbiOfHelpers<'tcx> {
    /// Compute a `FnAbi` suitable for indirect calls, i.e. to `fn` pointers.
    ///
    /// NB: this doesn't handle virtual calls - those should use `fn_abi_of_instance`
    /// instead, where the instance is an `InstanceDef::Virtual`.
    #[inline]
    fn fn_abi_of_fn_ptr(
        &self,
        sig: ty::PolyFnSig<'tcx>,
        extra_args: &'tcx ty::List<Ty<'tcx>>,
    ) -> Self::FnAbiOfResult {
        // FIXME(eddyb) get a better `span` here.
        let span = self.layout_tcx_at_span();
        let tcx = self.tcx().at(span);

        MaybeResult::from(tcx.fn_abi_of_fn_ptr(self.param_env().and((sig, extra_args))).map_err(
            |err| self.handle_fn_abi_err(err, span, FnAbiRequest::OfFnPtr { sig, extra_args }),
        ))
    }

    /// Compute a `FnAbi` suitable for declaring/defining an `fn` instance, and for
    /// direct calls to an `fn`.
    ///
    /// NB: that includes virtual calls, which are represented by "direct calls"
    /// to an `InstanceDef::Virtual` instance (of `<dyn Trait as Trait>::fn`).
    #[inline]
    fn fn_abi_of_instance(
        &self,
        instance: ty::Instance<'tcx>,
        extra_args: &'tcx ty::List<Ty<'tcx>>,
    ) -> Self::FnAbiOfResult {
        // FIXME(eddyb) get a better `span` here.
        let span = self.layout_tcx_at_span();
        let tcx = self.tcx().at(span);

        MaybeResult::from(
            tcx.fn_abi_of_instance(self.param_env().and((instance, extra_args))).map_err(|err| {
                // HACK(eddyb) at least for definitions of/calls to `Instance`s,
                // we can get some kind of span even if one wasn't provided.
                // However, we don't do this early in order to avoid calling
                // `def_span` unconditionally (which may have a perf penalty).
                let span = if !span.is_dummy() { span } else { tcx.def_span(instance.def_id()) };
                self.handle_fn_abi_err(err, span, FnAbiRequest::OfInstance { instance, extra_args })
            }),
        )
    }
}

impl<C: FnAbiOfHelpers<'tcx>> FnAbiOf<'tcx> for C {}

fn fn_abi_of_fn_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, FnAbiError<'tcx>> {
    let (param_env, (sig, extra_args)) = query.into_parts();

    LayoutCx { tcx, param_env }.fn_abi_new_uncached(
        sig,
        extra_args,
        None,
        CodegenFnAttrFlags::empty(),
        false,
    )
}

fn fn_abi_of_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, FnAbiError<'tcx>> {
    let (param_env, (instance, extra_args)) = query.into_parts();

    let sig = instance.fn_sig_for_fn_abi(tcx);

    let caller_location = if instance.def.requires_caller_location(tcx) {
        Some(tcx.caller_location_ty())
    } else {
        None
    };

    let attrs = tcx.codegen_fn_attrs(instance.def_id()).flags;

    LayoutCx { tcx, param_env }.fn_abi_new_uncached(
        sig,
        extra_args,
        caller_location,
        attrs,
        matches!(instance.def, ty::InstanceDef::Virtual(..)),
    )
}

impl<'tcx> LayoutCx<'tcx, TyCtxt<'tcx>> {
    // FIXME(eddyb) perhaps group the signature/type-containing (or all of them?)
    // arguments of this method, into a separate `struct`.
    fn fn_abi_new_uncached(
        &self,
        sig: ty::PolyFnSig<'tcx>,
        extra_args: &[Ty<'tcx>],
        caller_location: Option<Ty<'tcx>>,
        codegen_fn_attr_flags: CodegenFnAttrFlags,
        // FIXME(eddyb) replace this with something typed, like an `enum`.
        force_thin_self_ptr: bool,
    ) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, FnAbiError<'tcx>> {
        debug!("fn_abi_new_uncached({:?}, {:?})", sig, extra_args);

        let sig = self.tcx.normalize_erasing_late_bound_regions(self.param_env, sig);

        let conv = conv_from_spec_abi(self.tcx(), sig.abi);

        let mut inputs = sig.inputs();
        let extra_args = if sig.abi == RustCall {
            assert!(!sig.c_variadic && extra_args.is_empty());

            if let Some(input) = sig.inputs().last() {
                if let ty::Tuple(tupled_arguments) = input.kind() {
                    inputs = &sig.inputs()[0..sig.inputs().len() - 1];
                    tupled_arguments.iter().map(|k| k.expect_ty()).collect()
                } else {
                    bug!(
                        "argument to function with \"rust-call\" ABI \
                            is not a tuple"
                    );
                }
            } else {
                bug!(
                    "argument to function with \"rust-call\" ABI \
                        is not a tuple"
                );
            }
        } else {
            assert!(sig.c_variadic || extra_args.is_empty());
            extra_args.to_vec()
        };

        let target = &self.tcx.sess.target;
        let target_env_gnu_like = matches!(&target.env[..], "gnu" | "musl" | "uclibc");
        let win_x64_gnu = target.os == "windows" && target.arch == "x86_64" && target.env == "gnu";
        let linux_s390x_gnu_like =
            target.os == "linux" && target.arch == "s390x" && target_env_gnu_like;
        let linux_sparc64_gnu_like =
            target.os == "linux" && target.arch == "sparc64" && target_env_gnu_like;
        let linux_powerpc_gnu_like =
            target.os == "linux" && target.arch == "powerpc" && target_env_gnu_like;
        use SpecAbi::*;
        let rust_abi = matches!(sig.abi, RustIntrinsic | PlatformIntrinsic | Rust | RustCall);

        // Handle safe Rust thin and fat pointers.
        let adjust_for_rust_scalar = |attrs: &mut ArgAttributes,
                                      scalar: Scalar,
                                      layout: TyAndLayout<'tcx>,
                                      offset: Size,
                                      is_return: bool| {
            // Booleans are always an i1 that needs to be zero-extended.
            if scalar.is_bool() {
                attrs.ext(ArgExtension::Zext);
                return;
            }

            // Only pointer types handled below.
            if scalar.value != Pointer {
                return;
            }

            if !scalar.valid_range.contains(0) {
                attrs.set(ArgAttribute::NonNull);
            }

            if let Some(pointee) = layout.pointee_info_at(self, offset) {
                if let Some(kind) = pointee.safe {
                    attrs.pointee_align = Some(pointee.align);

                    // `Box` (`UniqueBorrowed`) are not necessarily dereferenceable
                    // for the entire duration of the function as they can be deallocated
                    // at any time. Set their valid size to 0.
                    attrs.pointee_size = match kind {
                        PointerKind::UniqueOwned => Size::ZERO,
                        _ => pointee.size,
                    };

                    // `Box` pointer parameters never alias because ownership is transferred
                    // `&mut` pointer parameters never alias other parameters,
                    // or mutable global data
                    //
                    // `&T` where `T` contains no `UnsafeCell<U>` is immutable,
                    // and can be marked as both `readonly` and `noalias`, as
                    // LLVM's definition of `noalias` is based solely on memory
                    // dependencies rather than pointer equality
                    //
                    // Due to past miscompiles in LLVM, we apply a separate NoAliasMutRef attribute
                    // for UniqueBorrowed arguments, so that the codegen backend can decide whether
                    // or not to actually emit the attribute. It can also be controlled with the
                    // `-Zmutable-noalias` debugging option.
                    let no_alias = match kind {
                        PointerKind::Shared | PointerKind::UniqueBorrowed => false,
                        PointerKind::UniqueOwned => true,
                        PointerKind::Frozen => !is_return,
                    };
                    if no_alias {
                        attrs.set(ArgAttribute::NoAlias);
                    }

                    if kind == PointerKind::Frozen && !is_return {
                        attrs.set(ArgAttribute::ReadOnly);
                    }

                    if kind == PointerKind::UniqueBorrowed && !is_return {
                        attrs.set(ArgAttribute::NoAliasMutRef);
                    }
                }
            }
        };

        let arg_of = |ty: Ty<'tcx>, arg_idx: Option<usize>| -> Result<_, FnAbiError<'tcx>> {
            let is_return = arg_idx.is_none();

            let layout = self.layout_of(ty)?;
            let layout = if force_thin_self_ptr && arg_idx == Some(0) {
                // Don't pass the vtable, it's not an argument of the virtual fn.
                // Instead, pass just the data pointer, but give it the type `*const/mut dyn Trait`
                // or `&/&mut dyn Trait` because this is special-cased elsewhere in codegen
                make_thin_self_ptr(self, layout)
            } else {
                layout
            };

            let mut arg = ArgAbi::new(self, layout, |layout, scalar, offset| {
                let mut attrs = ArgAttributes::new();
                adjust_for_rust_scalar(&mut attrs, scalar, *layout, offset, is_return);
                attrs
            });

            if arg.layout.is_zst() {
                // For some forsaken reason, x86_64-pc-windows-gnu
                // doesn't ignore zero-sized struct arguments.
                // The same is true for {s390x,sparc64,powerpc}-unknown-linux-{gnu,musl,uclibc}.
                if is_return
                    || rust_abi
                    || (!win_x64_gnu
                        && !linux_s390x_gnu_like
                        && !linux_sparc64_gnu_like
                        && !linux_powerpc_gnu_like)
                {
                    arg.mode = PassMode::Ignore;
                }
            }

            Ok(arg)
        };

        let mut fn_abi = FnAbi {
            ret: arg_of(sig.output(), None)?,
            args: inputs
                .iter()
                .cloned()
                .chain(extra_args)
                .chain(caller_location)
                .enumerate()
                .map(|(i, ty)| arg_of(ty, Some(i)))
                .collect::<Result<_, _>>()?,
            c_variadic: sig.c_variadic,
            fixed_count: inputs.len(),
            conv,
            can_unwind: fn_can_unwind(self.tcx(), codegen_fn_attr_flags, sig.abi),
        };
        self.fn_abi_adjust_for_abi(&mut fn_abi, sig.abi)?;
        debug!("fn_abi_new_uncached = {:?}", fn_abi);
        Ok(self.tcx.arena.alloc(fn_abi))
    }

    fn fn_abi_adjust_for_abi(
        &self,
        fn_abi: &mut FnAbi<'tcx, Ty<'tcx>>,
        abi: SpecAbi,
    ) -> Result<(), FnAbiError<'tcx>> {
        if abi == SpecAbi::Unadjusted {
            return Ok(());
        }

        if abi == SpecAbi::Rust
            || abi == SpecAbi::RustCall
            || abi == SpecAbi::RustIntrinsic
            || abi == SpecAbi::PlatformIntrinsic
        {
            let fixup = |arg: &mut ArgAbi<'tcx, Ty<'tcx>>| {
                if arg.is_ignore() {
                    return;
                }

                match arg.layout.abi {
                    Abi::Aggregate { .. } => {}

                    // This is a fun case! The gist of what this is doing is
                    // that we want callers and callees to always agree on the
                    // ABI of how they pass SIMD arguments. If we were to *not*
                    // make these arguments indirect then they'd be immediates
                    // in LLVM, which means that they'd used whatever the
                    // appropriate ABI is for the callee and the caller. That
                    // means, for example, if the caller doesn't have AVX
                    // enabled but the callee does, then passing an AVX argument
                    // across this boundary would cause corrupt data to show up.
                    //
                    // This problem is fixed by unconditionally passing SIMD
                    // arguments through memory between callers and callees
                    // which should get them all to agree on ABI regardless of
                    // target feature sets. Some more information about this
                    // issue can be found in #44367.
                    //
                    // Note that the platform intrinsic ABI is exempt here as
                    // that's how we connect up to LLVM and it's unstable
                    // anyway, we control all calls to it in libstd.
                    Abi::Vector { .. }
                        if abi != SpecAbi::PlatformIntrinsic
                            && self.tcx.sess.target.simd_types_indirect =>
                    {
                        arg.make_indirect();
                        return;
                    }

                    _ => return,
                }

                // Pass and return structures up to 2 pointers in size by value, matching `ScalarPair`.
                // LLVM will usually pass these in 2 registers, which is more efficient than by-ref.
                let max_by_val_size = Pointer.size(self) * 2;
                let size = arg.layout.size;

                if arg.layout.is_unsized() || size > max_by_val_size {
                    arg.make_indirect();
                } else {
                    // We want to pass small aggregates as immediates, but using
                    // a LLVM aggregate type for this leads to bad optimizations,
                    // so we pick an appropriately sized integer type instead.
                    arg.cast_to(Reg { kind: RegKind::Integer, size });
                }
            };
            fixup(&mut fn_abi.ret);
            for arg in &mut fn_abi.args {
                fixup(arg);
            }
        } else {
            fn_abi.adjust_for_foreign_abi(self, abi)?;
        }

        Ok(())
    }
}

fn make_thin_self_ptr<'tcx>(
    cx: &(impl HasTyCtxt<'tcx> + HasParamEnv<'tcx>),
    layout: TyAndLayout<'tcx>,
) -> TyAndLayout<'tcx> {
    let tcx = cx.tcx();
    let fat_pointer_ty = if layout.is_unsized() {
        // unsized `self` is passed as a pointer to `self`
        // FIXME (mikeyhew) change this to use &own if it is ever added to the language
        tcx.mk_mut_ptr(layout.ty)
    } else {
        match layout.abi {
            Abi::ScalarPair(..) => (),
            _ => bug!("receiver type has unsupported layout: {:?}", layout),
        }

        // In the case of Rc<Self>, we need to explicitly pass a *mut RcBox<Self>
        // with a Scalar (not ScalarPair) ABI. This is a hack that is understood
        // elsewhere in the compiler as a method on a `dyn Trait`.
        // To get the type `*mut RcBox<Self>`, we just keep unwrapping newtypes until we
        // get a built-in pointer type
        let mut fat_pointer_layout = layout;
        'descend_newtypes: while !fat_pointer_layout.ty.is_unsafe_ptr()
            && !fat_pointer_layout.ty.is_region_ptr()
        {
            for i in 0..fat_pointer_layout.fields.count() {
                let field_layout = fat_pointer_layout.field(cx, i);

                if !field_layout.is_zst() {
                    fat_pointer_layout = field_layout;
                    continue 'descend_newtypes;
                }
            }

            bug!("receiver has no non-zero-sized fields {:?}", fat_pointer_layout);
        }

        fat_pointer_layout.ty
    };

    // we now have a type like `*mut RcBox<dyn Trait>`
    // change its layout to that of `*mut ()`, a thin pointer, but keep the same type
    // this is understood as a special case elsewhere in the compiler
    let unit_ptr_ty = tcx.mk_mut_ptr(tcx.mk_unit());

    TyAndLayout {
        ty: fat_pointer_ty,

        // NOTE(eddyb) using an empty `ParamEnv`, and `unwrap`-ing the `Result`
        // should always work because the type is always `*mut ()`.
        ..tcx.layout_of(ty::ParamEnv::reveal_all().and(unit_ptr_ty)).unwrap()
    }
}
