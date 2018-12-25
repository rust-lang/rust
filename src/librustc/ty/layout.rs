use session::{self, DataTypeKind};
use ty::{self, Ty, TyCtxt, TypeFoldable, ReprOptions};

use syntax::ast::{self, IntTy, UintTy};
use syntax::attr;
use syntax_pos::DUMMY_SP;

use std::cmp;
use std::fmt;
use std::i128;
use std::iter;
use std::mem;
use std::ops::Bound;

use ich::StableHashingContext;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};

pub use rustc_target::abi::*;

pub trait IntegerExt {
    fn to_ty<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, signed: bool) -> Ty<'tcx>;
    fn from_attr<C: HasDataLayout>(cx: &C, ity: attr::IntType) -> Integer;
    fn repr_discr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            ty: Ty<'tcx>,
                            repr: &ReprOptions,
                            min: i128,
                            max: i128)
                            -> (Integer, bool);
}

impl IntegerExt for Integer {
    fn to_ty<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, signed: bool) -> Ty<'tcx> {
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

    /// Get the Integer type from an attr::IntType.
    fn from_attr<C: HasDataLayout>(cx: &C, ity: attr::IntType) -> Integer {
        let dl = cx.data_layout();

        match ity {
            attr::SignedInt(IntTy::I8) | attr::UnsignedInt(UintTy::U8) => I8,
            attr::SignedInt(IntTy::I16) | attr::UnsignedInt(UintTy::U16) => I16,
            attr::SignedInt(IntTy::I32) | attr::UnsignedInt(UintTy::U32) => I32,
            attr::SignedInt(IntTy::I64) | attr::UnsignedInt(UintTy::U64) => I64,
            attr::SignedInt(IntTy::I128) | attr::UnsignedInt(UintTy::U128) => I128,
            attr::SignedInt(IntTy::Isize) | attr::UnsignedInt(UintTy::Usize) => {
                dl.ptr_sized_integer()
            }
        }
    }

    /// Find the appropriate Integer type and signedness for the given
    /// signed discriminant range and #[repr] attribute.
    /// N.B.: u128 values above i128::MAX will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            ty: Ty<'tcx>,
                            repr: &ReprOptions,
                            min: i128,
                            max: i128)
                            -> (Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u128, max as u128));
        let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        let mut min_from_extern = None;
        let min_default = I8;

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(&tcx, ity);
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                bug!("Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum `{}", ty)
            }
            return (discr, ity.is_signed());
        }

        if repr.c() {
            match &tcx.sess.target.target.arch[..] {
                // WARNING: the ARM EABI has two variants; the one corresponding
                // to `at_least == I32` appears to be used on Linux and NetBSD,
                // but some systems may use the variant corresponding to no
                // lower bound. However, we don't run on those yet...?
                "arm" => min_from_extern = Some(I32),
                _ => min_from_extern = Some(I32),
            }
        }

        let at_least = min_from_extern.unwrap_or(min_default);

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (cmp::max(unsigned_fit, at_least), false)
        } else {
            (cmp::max(signed_fit, at_least), true)
        }
    }
}

pub trait PrimitiveExt {
    fn to_ty<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx>;
}

impl PrimitiveExt for Primitive {
    fn to_ty<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            Int(i, signed) => i.to_ty(tcx, signed),
            Float(FloatTy::F32) => tcx.types.f32,
            Float(FloatTy::F64) => tcx.types.f64,
            Pointer => tcx.mk_mut_ptr(tcx.mk_unit()),
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

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum LayoutError<'tcx> {
    Unknown(Ty<'tcx>),
    SizeOverflow(Ty<'tcx>)
}

impl<'tcx> fmt::Display for LayoutError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            LayoutError::Unknown(ty) => {
                write!(f, "the type `{:?}` has an unknown layout", ty)
            }
            LayoutError::SizeOverflow(ty) => {
                write!(f, "the type `{:?}` is too big for the current architecture", ty)
            }
        }
    }
}

fn layout_raw<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                        -> Result<&'tcx LayoutDetails, LayoutError<'tcx>>
{
    ty::tls::with_related_context(tcx, move |icx| {
        let rec_limit = *tcx.sess.recursion_limit.get();
        let (param_env, ty) = query.into_parts();

        if icx.layout_depth > rec_limit {
            tcx.sess.fatal(
                &format!("overflow representing the type `{}`", ty));
        }

        // Update the ImplicitCtxt to increase the layout_depth
        let icx = ty::tls::ImplicitCtxt {
            layout_depth: icx.layout_depth + 1,
            ..icx.clone()
        };

        ty::tls::enter_context(&icx, |_| {
            let cx = LayoutCx { tcx, param_env };
            let layout = cx.layout_raw_uncached(ty);
            // Type-level uninhabitedness should always imply ABI uninhabitedness.
            if let Ok(layout) = layout {
                if ty.conservative_is_privately_uninhabited(tcx) {
                    assert!(layout.abi.is_uninhabited());
                }
            }
            layout
        })
    })
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    *providers = ty::query::Providers {
        layout_raw,
        ..*providers
    };
}

pub struct LayoutCx<'tcx, C> {
    pub tcx: C,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'a, 'tcx> LayoutCx<'tcx, TyCtxt<'a, 'tcx, 'tcx>> {
    fn layout_raw_uncached(&self, ty: Ty<'tcx>) -> Result<&'tcx LayoutDetails, LayoutError<'tcx>> {
        let tcx = self.tcx;
        let param_env = self.param_env;
        let dl = self.data_layout();
        let scalar_unit = |value: Primitive| {
            let bits = value.size(dl).bits();
            assert!(bits <= 128);
            Scalar {
                value,
                valid_range: 0..=(!0 >> (128 - bits))
            }
        };
        let scalar = |value: Primitive| {
            tcx.intern_layout(LayoutDetails::scalar(self, scalar_unit(value)))
        };
        let scalar_pair = |a: Scalar, b: Scalar| {
            let b_align = b.value.align(dl);
            let align = a.value.align(dl).max(b_align).max(dl.aggregate_align);
            let b_offset = a.value.size(dl).align_to(b_align.abi);
            let size = (b_offset + b.value.size(dl)).align_to(align.abi);
            LayoutDetails {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields: FieldPlacement::Arbitrary {
                    offsets: vec![Size::ZERO, b_offset],
                    memory_index: vec![0, 1]
                },
                abi: Abi::ScalarPair(a, b),
                align,
                size
            }
        };

        #[derive(Copy, Clone, Debug)]
        enum StructKind {
            /// A tuple, closure, or univariant which cannot be coerced to unsized.
            AlwaysSized,
            /// A univariant, the last field of which may be coerced to unsized.
            MaybeUnsized,
            /// A univariant, but with a prefix of an arbitrary size & alignment (e.g., enum tag).
            Prefixed(Size, Align),
        }

        let univariant_uninterned = |fields: &[TyLayout<'_>], repr: &ReprOptions, kind| {
            let packed = repr.packed();
            if packed && repr.align > 0 {
                bug!("struct cannot be packed and aligned");
            }

            let pack = Align::from_bytes(repr.pack as u64).unwrap();

            let mut align = if packed {
                dl.i8_align
            } else {
                dl.aggregate_align
            };

            let mut sized = true;
            let mut offsets = vec![Size::ZERO; fields.len()];
            let mut inverse_memory_index: Vec<u32> = (0..fields.len() as u32).collect();

            let mut optimize = !repr.inhibit_struct_field_reordering_opt();
            if let StructKind::Prefixed(_, align) = kind {
                optimize &= align.bytes() == 1;
            }

            if optimize {
                let end = if let StructKind::MaybeUnsized = kind {
                    fields.len() - 1
                } else {
                    fields.len()
                };
                let optimizing = &mut inverse_memory_index[..end];
                let field_align = |f: &TyLayout<'_>| {
                    if packed { f.align.abi.min(pack) } else { f.align.abi }
                };
                match kind {
                    StructKind::AlwaysSized |
                    StructKind::MaybeUnsized => {
                        optimizing.sort_by_key(|&x| {
                            // Place ZSTs first to avoid "interesting offsets",
                            // especially with only one or two non-ZST fields.
                            let f = &fields[x as usize];
                            (!f.is_zst(), cmp::Reverse(field_align(f)))
                        });
                    }
                    StructKind::Prefixed(..) => {
                        optimizing.sort_by_key(|&x| field_align(&fields[x as usize]));
                    }
                }
            }

            // inverse_memory_index holds field indices by increasing memory offset.
            // That is, if field 5 has offset 0, the first element of inverse_memory_index is 5.
            // We now write field offsets to the corresponding offset slot;
            // field 5 with offset 0 puts 0 in offsets[5].
            // At the bottom of this function, we use inverse_memory_index to produce memory_index.

            let mut offset = Size::ZERO;

            if let StructKind::Prefixed(prefix_size, prefix_align) = kind {
                let prefix_align = if packed {
                    prefix_align.min(pack)
                } else {
                    prefix_align
                };
                align = align.max(AbiAndPrefAlign::new(prefix_align));
                offset = prefix_size.align_to(prefix_align);
            }

            for &i in &inverse_memory_index {
                let field = fields[i as usize];
                if !sized {
                    bug!("univariant: field #{} of `{}` comes after unsized field",
                         offsets.len(), ty);
                }

                if field.is_unsized() {
                    sized = false;
                }

                // Invariant: offset < dl.obj_size_bound() <= 1<<61
                let field_align = if packed {
                    field.align.min(AbiAndPrefAlign::new(pack))
                } else {
                    field.align
                };
                offset = offset.align_to(field_align.abi);
                align = align.max(field_align);

                debug!("univariant offset: {:?} field: {:#?}", offset, field);
                offsets[i as usize] = offset;

                offset = offset.checked_add(field.size, dl)
                    .ok_or(LayoutError::SizeOverflow(ty))?;
            }

            if repr.align > 0 {
                let repr_align = repr.align as u64;
                align = align.max(AbiAndPrefAlign::new(Align::from_bytes(repr_align).unwrap()));
                debug!("univariant repr_align: {:?}", repr_align);
            }

            debug!("univariant min_size: {:?}", offset);
            let min_size = offset;

            // As stated above, inverse_memory_index holds field indices by increasing offset.
            // This makes it an already-sorted view of the offsets vec.
            // To invert it, consider:
            // If field 5 has offset 0, offsets[0] is 5, and memory_index[5] should be 0.
            // Field 5 would be the first element, so memory_index is i:
            // Note: if we didn't optimize, it's already right.

            let mut memory_index;
            if optimize {
                memory_index = vec![0; inverse_memory_index.len()];

                for i in 0..inverse_memory_index.len() {
                    memory_index[inverse_memory_index[i] as usize]  = i as u32;
                }
            } else {
                memory_index = inverse_memory_index;
            }

            let size = min_size.align_to(align.abi);
            let mut abi = Abi::Aggregate { sized };

            // Unpack newtype ABIs and find scalar pairs.
            if sized && size.bytes() > 0 {
                // All other fields must be ZSTs, and we need them to all start at 0.
                let mut zst_offsets =
                    offsets.iter().enumerate().filter(|&(i, _)| fields[i].is_zst());
                if zst_offsets.all(|(_, o)| o.bytes() == 0) {
                    let mut non_zst_fields =
                        fields.iter().enumerate().filter(|&(_, f)| !f.is_zst());

                    match (non_zst_fields.next(), non_zst_fields.next(), non_zst_fields.next()) {
                        // We have exactly one non-ZST field.
                        (Some((i, field)), None, None) => {
                            // Field fills the struct and it has a scalar or scalar pair ABI.
                            if offsets[i].bytes() == 0 &&
                               align.abi == field.align.abi &&
                               size == field.size {
                                match field.abi {
                                    // For plain scalars, or vectors of them, we can't unpack
                                    // newtypes for `#[repr(C)]`, as that affects C ABIs.
                                    Abi::Scalar(_) | Abi::Vector { .. } if optimize => {
                                        abi = field.abi.clone();
                                    }
                                    // But scalar pairs are Rust-specific and get
                                    // treated as aggregates by C ABIs anyway.
                                    Abi::ScalarPair(..) => {
                                        abi = field.abi.clone();
                                    }
                                    _ => {}
                                }
                            }
                        }

                        // Two non-ZST fields, and they're both scalars.
                        (Some((i, &TyLayout {
                            details: &LayoutDetails { abi: Abi::Scalar(ref a), .. }, ..
                        })), Some((j, &TyLayout {
                            details: &LayoutDetails { abi: Abi::Scalar(ref b), .. }, ..
                        })), None) => {
                            // Order by the memory placement, not source order.
                            let ((i, a), (j, b)) = if offsets[i] < offsets[j] {
                                ((i, a), (j, b))
                            } else {
                                ((j, b), (i, a))
                            };
                            let pair = scalar_pair(a.clone(), b.clone());
                            let pair_offsets = match pair.fields {
                                FieldPlacement::Arbitrary {
                                    ref offsets,
                                    ref memory_index
                                } => {
                                    assert_eq!(memory_index, &[0, 1]);
                                    offsets
                                }
                                _ => bug!()
                            };
                            if offsets[i] == pair_offsets[0] &&
                               offsets[j] == pair_offsets[1] &&
                               align == pair.align &&
                               size == pair.size {
                                // We can use `ScalarPair` only when it matches our
                                // already computed layout (including `#[repr(C)]`).
                                abi = pair.abi;
                            }
                        }

                        _ => {}
                    }
                }
            }

            if sized && fields.iter().any(|f| f.abi.is_uninhabited()) {
                abi = Abi::Uninhabited;
            }

            Ok(LayoutDetails {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields: FieldPlacement::Arbitrary {
                    offsets,
                    memory_index
                },
                abi,
                align,
                size
            })
        };
        let univariant = |fields: &[TyLayout<'_>], repr: &ReprOptions, kind| {
            Ok(tcx.intern_layout(univariant_uninterned(fields, repr, kind)?))
        };
        debug_assert!(!ty.has_infer_types());

        Ok(match ty.sty {
            // Basic scalars.
            ty::Bool => {
                tcx.intern_layout(LayoutDetails::scalar(self, Scalar {
                    value: Int(I8, false),
                    valid_range: 0..=1
                }))
            }
            ty::Char => {
                tcx.intern_layout(LayoutDetails::scalar(self, Scalar {
                    value: Int(I32, false),
                    valid_range: 0..=0x10FFFF
                }))
            }
            ty::Int(ity) => {
                scalar(Int(Integer::from_attr(dl, attr::SignedInt(ity)), true))
            }
            ty::Uint(ity) => {
                scalar(Int(Integer::from_attr(dl, attr::UnsignedInt(ity)), false))
            }
            ty::Float(fty) => scalar(Float(fty)),
            ty::FnPtr(_) => {
                let mut ptr = scalar_unit(Pointer);
                ptr.valid_range = 1..=*ptr.valid_range.end();
                tcx.intern_layout(LayoutDetails::scalar(self, ptr))
            }

            // The never type.
            ty::Never => {
                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldPlacement::Union(0),
                    abi: Abi::Uninhabited,
                    align: dl.i8_align,
                    size: Size::ZERO
                })
            }

            // Potentially-fat pointers.
            ty::Ref(_, pointee, _) |
            ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                let mut data_ptr = scalar_unit(Pointer);
                if !ty.is_unsafe_ptr() {
                    data_ptr.valid_range = 1..=*data_ptr.valid_range.end();
                }

                let pointee = tcx.normalize_erasing_regions(param_env, pointee);
                if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
                    return Ok(tcx.intern_layout(LayoutDetails::scalar(self, data_ptr)));
                }

                let unsized_part = tcx.struct_tail(pointee);
                let metadata = match unsized_part.sty {
                    ty::Foreign(..) => {
                        return Ok(tcx.intern_layout(LayoutDetails::scalar(self, data_ptr)));
                    }
                    ty::Slice(_) | ty::Str => {
                        scalar_unit(Int(dl.ptr_sized_integer(), false))
                    }
                    ty::Dynamic(..) => {
                        let mut vtable = scalar_unit(Pointer);
                        vtable.valid_range = 1..=*vtable.valid_range.end();
                        vtable
                    }
                    _ => return Err(LayoutError::Unknown(unsized_part))
                };

                // Effectively a (ptr, meta) tuple.
                tcx.intern_layout(scalar_pair(data_ptr, metadata))
            }

            // Arrays and slices.
            ty::Array(element, mut count) => {
                if count.has_projections() {
                    count = tcx.normalize_erasing_regions(param_env, count);
                    if count.has_projections() {
                        return Err(LayoutError::Unknown(ty));
                    }
                }

                let element = self.layout_of(element)?;
                let count = count.unwrap_usize(tcx);
                let size = element.size.checked_mul(count, dl)
                    .ok_or(LayoutError::SizeOverflow(ty))?;

                let abi = if count != 0 && ty.conservative_is_privately_uninhabited(tcx) {
                    Abi::Uninhabited
                } else {
                    Abi::Aggregate { sized: true }
                };

                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldPlacement::Array {
                        stride: element.size,
                        count
                    },
                    abi,
                    align: element.align,
                    size
                })
            }
            ty::Slice(element) => {
                let element = self.layout_of(element)?;
                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldPlacement::Array {
                        stride: element.size,
                        count: 0
                    },
                    abi: Abi::Aggregate { sized: false },
                    align: element.align,
                    size: Size::ZERO
                })
            }
            ty::Str => {
                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldPlacement::Array {
                        stride: Size::from_bytes(1),
                        count: 0
                    },
                    abi: Abi::Aggregate { sized: false },
                    align: dl.i8_align,
                    size: Size::ZERO
                })
            }

            // Odd unit types.
            ty::FnDef(..) => {
                univariant(&[], &ReprOptions::default(), StructKind::AlwaysSized)?
            }
            ty::Dynamic(..) | ty::Foreign(..) => {
                let mut unit = univariant_uninterned(&[], &ReprOptions::default(),
                  StructKind::AlwaysSized)?;
                match unit.abi {
                    Abi::Aggregate { ref mut sized } => *sized = false,
                    _ => bug!()
                }
                tcx.intern_layout(unit)
            }

            // Tuples, generators and closures.
            ty::Generator(def_id, ref substs, _) => {
                let tys = substs.field_tys(def_id, tcx);
                univariant(&tys.map(|ty| self.layout_of(ty)).collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    StructKind::AlwaysSized)?
            }

            ty::Closure(def_id, ref substs) => {
                let tys = substs.upvar_tys(def_id, tcx);
                univariant(&tys.map(|ty| self.layout_of(ty)).collect::<Result<Vec<_>, _>>()?,
                    &ReprOptions::default(),
                    StructKind::AlwaysSized)?
            }

            ty::Tuple(tys) => {
                let kind = if tys.len() == 0 {
                    StructKind::AlwaysSized
                } else {
                    StructKind::MaybeUnsized
                };

                univariant(&tys.iter().map(|ty| self.layout_of(ty)).collect::<Result<Vec<_>, _>>()?,
                           &ReprOptions::default(), kind)?
            }

            // SIMD vector types.
            ty::Adt(def, ..) if def.repr.simd() => {
                let element = self.layout_of(ty.simd_type(tcx))?;
                let count = ty.simd_size(tcx) as u64;
                assert!(count > 0);
                let scalar = match element.abi {
                    Abi::Scalar(ref scalar) => scalar.clone(),
                    _ => {
                        tcx.sess.fatal(&format!("monomorphising SIMD type `{}` with \
                                                 a non-machine element type `{}`",
                                                ty, element.ty));
                    }
                };
                let size = element.size.checked_mul(count, dl)
                    .ok_or(LayoutError::SizeOverflow(ty))?;
                let align = dl.vector_align(size);
                let size = size.align_to(align.abi);

                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Single { index: VariantIdx::new(0) },
                    fields: FieldPlacement::Array {
                        stride: element.size,
                        count
                    },
                    abi: Abi::Vector {
                        element: scalar,
                        count
                    },
                    size,
                    align,
                })
            }

            // ADTs.
            ty::Adt(def, substs) => {
                // Cache the field layouts.
                let variants = def.variants.iter().map(|v| {
                    v.fields.iter().map(|field| {
                        self.layout_of(field.ty(tcx, substs))
                    }).collect::<Result<Vec<_>, _>>()
                }).collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

                if def.is_union() {
                    let packed = def.repr.packed();
                    if packed && def.repr.align > 0 {
                        bug!("Union cannot be packed and aligned");
                    }

                    let pack = Align::from_bytes(def.repr.pack as u64).unwrap();

                    let mut align = if packed {
                        dl.i8_align
                    } else {
                        dl.aggregate_align
                    };

                    if def.repr.align > 0 {
                        let repr_align = def.repr.align as u64;
                        align = align.max(
                            AbiAndPrefAlign::new(Align::from_bytes(repr_align).unwrap()));
                    }

                    let optimize = !def.repr.inhibit_union_abi_opt();
                    let mut size = Size::ZERO;
                    let mut abi = Abi::Aggregate { sized: true };
                    let index = VariantIdx::new(0);
                    for field in &variants[index] {
                        assert!(!field.is_unsized());

                        let field_align = if packed {
                            field.align.min(AbiAndPrefAlign::new(pack))
                        } else {
                            field.align
                        };
                        align = align.max(field_align);

                        // If all non-ZST fields have the same ABI, forward this ABI
                        if optimize && !field.is_zst() {
                            // Normalize scalar_unit to the maximal valid range
                            let field_abi = match &field.abi {
                                Abi::Scalar(x) => Abi::Scalar(scalar_unit(x.value)),
                                Abi::ScalarPair(x, y) => {
                                    Abi::ScalarPair(
                                        scalar_unit(x.value),
                                        scalar_unit(y.value),
                                    )
                                }
                                Abi::Vector { element: x, count } => {
                                    Abi::Vector {
                                        element: scalar_unit(x.value),
                                        count: *count,
                                    }
                                }
                                Abi::Uninhabited |
                                Abi::Aggregate { .. }  => Abi::Aggregate { sized: true },
                            };

                            if size == Size::ZERO {
                                // first non ZST: initialize 'abi'
                                abi = field_abi;
                            } else if abi != field_abi  {
                                // different fields have different ABI: reset to Aggregate
                                abi = Abi::Aggregate { sized: true };
                            }
                        }

                        size = cmp::max(size, field.size);
                    }

                    return Ok(tcx.intern_layout(LayoutDetails {
                        variants: Variants::Single { index },
                        fields: FieldPlacement::Union(variants[index].len()),
                        abi,
                        align,
                        size: size.align_to(align.abi)
                    }));
                }

                // A variant is absent if it's uninhabited and only has ZST fields.
                // Present uninhabited variants only require space for their fields,
                // but *not* an encoding of the discriminant (e.g., a tag value).
                // See issue #49298 for more details on the need to leave space
                // for non-ZST uninhabited data (mostly partial initialization).
                let absent = |fields: &[TyLayout<'_>]| {
                    let uninhabited = fields.iter().any(|f| f.abi.is_uninhabited());
                    let is_zst = fields.iter().all(|f| f.is_zst());
                    uninhabited && is_zst
                };
                let (present_first, present_second) = {
                    let mut present_variants = variants.iter_enumerated().filter_map(|(i, v)| {
                        if absent(v) {
                            None
                        } else {
                            Some(i)
                        }
                    });
                    (present_variants.next(), present_variants.next())
                };
                if present_first.is_none() {
                    // Uninhabited because it has no variants, or only absent ones.
                    return tcx.layout_raw(param_env.and(tcx.types.never));
                }

                let is_struct = !def.is_enum() ||
                    // Only one variant is present.
                    (present_second.is_none() &&
                    // Representation optimizations are allowed.
                    !def.repr.inhibit_enum_layout_opt());
                if is_struct {
                    // Struct, or univariant enum equivalent to a struct.
                    // (Typechecking will reject discriminant-sizing attrs.)

                    let v = present_first.unwrap();
                    let kind = if def.is_enum() || variants[v].len() == 0 {
                        StructKind::AlwaysSized
                    } else {
                        let param_env = tcx.param_env(def.did);
                        let last_field = def.variants[v].fields.last().unwrap();
                        let always_sized = tcx.type_of(last_field.did)
                                              .is_sized(tcx.at(DUMMY_SP), param_env);
                        if !always_sized { StructKind::MaybeUnsized }
                        else { StructKind::AlwaysSized }
                    };

                    let mut st = univariant_uninterned(&variants[v], &def.repr, kind)?;
                    st.variants = Variants::Single { index: v };
                    let (start, end) = self.tcx.layout_scalar_valid_range(def.did);
                    match st.abi {
                        Abi::Scalar(ref mut scalar) |
                        Abi::ScalarPair(ref mut scalar, _) => {
                            // the asserts ensure that we are not using the
                            // `#[rustc_layout_scalar_valid_range(n)]`
                            // attribute to widen the range of anything as that would probably
                            // result in UB somewhere
                            if let Bound::Included(start) = start {
                                assert!(*scalar.valid_range.start() <= start);
                                scalar.valid_range = start..=*scalar.valid_range.end();
                            }
                            if let Bound::Included(end) = end {
                                assert!(*scalar.valid_range.end() >= end);
                                scalar.valid_range = *scalar.valid_range.start()..=end;
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

                // The current code for niche-filling relies on variant indices
                // instead of actual discriminants, so dataful enums with
                // explicit discriminants (RFC #2363) would misbehave.
                let no_explicit_discriminants = def.variants.iter_enumerated()
                    .all(|(i, v)| v.discr == ty::VariantDiscr::Relative(i.as_u32()));

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
                        let count = (
                            niche_variants.end().as_u32() - niche_variants.start().as_u32() + 1
                        ) as u128;
                        for (field_index, &field) in variants[i].iter().enumerate() {
                            let niche = match self.find_niche(field)? {
                                Some(niche) => niche,
                                _ => continue,
                            };
                            let (niche_start, niche_scalar) = match niche.reserve(self, count) {
                                Some(pair) => pair,
                                None => continue,
                            };

                            let mut align = dl.aggregate_align;
                            let st = variants.iter_enumerated().map(|(j, v)| {
                                let mut st = univariant_uninterned(v,
                                    &def.repr, StructKind::AlwaysSized)?;
                                st.variants = Variants::Single { index: j };

                                align = align.max(st.align);

                                Ok(st)
                            }).collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

                            let offset = st[i].fields.offset(field_index) + niche.offset;
                            let size = st[i].size;

                            let mut abi = match st[i].abi {
                                Abi::Scalar(_) => Abi::Scalar(niche_scalar.clone()),
                                Abi::ScalarPair(ref first, ref second) => {
                                    // We need to use scalar_unit to reset the
                                    // valid range to the maximal one for that
                                    // primitive, because only the niche is
                                    // guaranteed to be initialised, not the
                                    // other primitive.
                                    if offset.bytes() == 0 {
                                        Abi::ScalarPair(
                                            niche_scalar.clone(),
                                            scalar_unit(second.value),
                                        )
                                    } else {
                                        Abi::ScalarPair(
                                            scalar_unit(first.value),
                                            niche_scalar.clone(),
                                        )
                                    }
                                }
                                _ => Abi::Aggregate { sized: true },
                            };

                            if st.iter().all(|v| v.abi.is_uninhabited()) {
                                abi = Abi::Uninhabited;
                            }

                            return Ok(tcx.intern_layout(LayoutDetails {
                                variants: Variants::NicheFilling {
                                    dataful_variant: i,
                                    niche_variants,
                                    niche: niche_scalar,
                                    niche_start,
                                    variants: st,
                                },
                                fields: FieldPlacement::Arbitrary {
                                    offsets: vec![offset],
                                    memory_index: vec![0]
                                },
                                abi,
                                size,
                                align,
                            }));
                        }
                    }
                }

                let (mut min, mut max) = (i128::max_value(), i128::min_value());
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
                    if x < min { min = x; }
                    if x > max { max = x; }
                }
                // We might have no inhabited variants, so pretend there's at least one.
                if (min, max) == (i128::max_value(), i128::min_value()) {
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
                let mut layout_variants = variants.iter_enumerated().map(|(i, field_layouts)| {
                    let mut st = univariant_uninterned(&field_layouts,
                        &def.repr, StructKind::Prefixed(min_ity.size(), prefix_align))?;
                    st.variants = Variants::Single { index: i };
                    // Find the first field we can't move later
                    // to make room for a larger discriminant.
                    for field in st.fields.index_by_increasing_offset().map(|j| field_layouts[j]) {
                        if !field.is_zst() || field.align.abi.bytes() != 1 {
                            start_align = start_align.min(field.align.abi);
                            break;
                        }
                    }
                    size = cmp::max(size, st.size);
                    align = align.max(st.align);
                    Ok(st)
                }).collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

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
                    bug!("layout decided on a larger discriminant type ({:?}) than typeck ({:?})",
                         min_ity, typeck_ity);
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
                            FieldPlacement::Arbitrary { ref mut offsets, .. } => {
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
                            _ => bug!()
                        }
                    }
                }

                let tag_mask = !0u128 >> (128 - ity.size().bits());
                let tag = Scalar {
                    value: Int(ity, signed),
                    valid_range: (min as u128 & tag_mask)..=(max as u128 & tag_mask),
                };
                let mut abi = Abi::Aggregate { sized: true };
                if tag.value.size(dl) == size {
                    abi = Abi::Scalar(tag.clone());
                } else {
                    // Try to use a ScalarPair for all tagged enums.
                    let mut common_prim = None;
                    for (field_layouts, layout_variant) in variants.iter().zip(&layout_variants) {
                        let offsets = match layout_variant.fields {
                            FieldPlacement::Arbitrary { ref offsets, .. } => offsets,
                            _ => bug!(),
                        };
                        let mut fields = field_layouts
                            .iter()
                            .zip(offsets)
                            .filter(|p| !p.0.is_zst());
                        let (field, offset) = match (fields.next(), fields.next()) {
                            (None, None) => continue,
                            (Some(pair), None) => pair,
                            _ => {
                                common_prim = None;
                                break;
                            }
                        };
                        let prim = match field.details.abi {
                            Abi::Scalar(ref scalar) => scalar.value,
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
                        let pair = scalar_pair(tag.clone(), scalar_unit(prim));
                        let pair_offsets = match pair.fields {
                            FieldPlacement::Arbitrary {
                                ref offsets,
                                ref memory_index
                            } => {
                                assert_eq!(memory_index, &[0, 1]);
                                offsets
                            }
                            _ => bug!()
                        };
                        if pair_offsets[0] == Size::ZERO &&
                            pair_offsets[1] == *offset &&
                            align == pair.align &&
                            size == pair.size {
                            // We can use `ScalarPair` only when it matches our
                            // already computed layout (including `#[repr(C)]`).
                            abi = pair.abi;
                        }
                    }
                }

                if layout_variants.iter().all(|v| v.abi.is_uninhabited()) {
                    abi = Abi::Uninhabited;
                }

                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Tagged {
                        tag,
                        variants: layout_variants,
                    },
                    fields: FieldPlacement::Arbitrary {
                        offsets: vec![Size::ZERO],
                        memory_index: vec![0]
                    },
                    abi,
                    align,
                    size
                })
            }

            // Types with no meaningful known layout.
            ty::Projection(_) | ty::Opaque(..) => {
                let normalized = tcx.normalize_erasing_regions(param_env, ty);
                if ty == normalized {
                    return Err(LayoutError::Unknown(ty));
                }
                tcx.layout_raw(param_env.and(normalized))?
            }

            ty::Bound(..) |
            ty::Placeholder(..) |
            ty::UnnormalizedProjection(..) |
            ty::GeneratorWitness(..) |
            ty::Infer(_) => {
                bug!("LayoutDetails::compute: unexpected type `{}`", ty)
            }

            ty::Param(_) | ty::Error => {
                return Err(LayoutError::Unknown(ty));
            }
        })
    }

    /// This is invoked by the `layout_raw` query to record the final
    /// layout of each type.
    #[inline]
    fn record_layout_for_printing(&self, layout: TyLayout<'tcx>) {
        // If we are running with `-Zprint-type-sizes`, record layouts for
        // dumping later. Ignore layouts that are done with non-empty
        // environments or non-monomorphic layouts, as the user only wants
        // to see the stuff resulting from the final codegen session.
        if
            !self.tcx.sess.opts.debugging_opts.print_type_sizes ||
            layout.ty.has_param_types() ||
            layout.ty.has_self_ty() ||
            !self.param_env.caller_bounds.is_empty()
        {
            return;
        }

        self.record_layout_for_printing_outlined(layout)
    }

    fn record_layout_for_printing_outlined(&self, layout: TyLayout<'tcx>) {
        // (delay format until we actually need it)
        let record = |kind, packed, opt_discr_size, variants| {
            let type_desc = format!("{:?}", layout.ty);
            self.tcx.sess.code_stats.borrow_mut().record_type_size(kind,
                                                                   type_desc,
                                                                   layout.align.abi,
                                                                   layout.size,
                                                                   packed,
                                                                   opt_discr_size,
                                                                   variants);
        };

        let adt_def = match layout.ty.sty {
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
        let adt_packed = adt_def.repr.packed();

        let build_variant_info = |n: Option<ast::Name>,
                                  flds: &[ast::Name],
                                  layout: TyLayout<'tcx>| {
            let mut min_size = Size::ZERO;
            let field_info: Vec<_> = flds.iter().enumerate().map(|(i, &name)| {
                match layout.field(self, i) {
                    Err(err) => {
                        bug!("no layout found for field {}: `{:?}`", name, err);
                    }
                    Ok(field_layout) => {
                        let offset = layout.fields.offset(i);
                        let field_end = offset + field_layout.size;
                        if min_size < field_end {
                            min_size = field_end;
                        }
                        session::FieldInfo {
                            name: name.to_string(),
                            offset: offset.bytes(),
                            size: field_layout.size.bytes(),
                            align: field_layout.align.abi.bytes(),
                        }
                    }
                }
            }).collect();

            session::VariantInfo {
                name: n.map(|n| n.to_string()),
                kind: if layout.is_unsized() {
                    session::SizeKind::Min
                } else {
                    session::SizeKind::Exact
                },
                align: layout.align.abi.bytes(),
                size: if min_size.bytes() == 0 {
                    layout.size.bytes()
                } else {
                    min_size.bytes()
                },
                fields: field_info,
            }
        };

        match layout.variants {
            Variants::Single { index } => {
                debug!("print-type-size `{:#?}` variant {}",
                       layout, adt_def.variants[index].name);
                if !adt_def.variants.is_empty() {
                    let variant_def = &adt_def.variants[index];
                    let fields: Vec<_> =
                        variant_def.fields.iter().map(|f| f.ident.name).collect();
                    record(adt_kind.into(),
                           adt_packed,
                           None,
                           vec![build_variant_info(Some(variant_def.name),
                                                   &fields,
                                                   layout)]);
                } else {
                    // (This case arises for *empty* enums; so give it
                    // zero variants.)
                    record(adt_kind.into(), adt_packed, None, vec![]);
                }
            }

            Variants::NicheFilling { .. } |
            Variants::Tagged { .. } => {
                debug!("print-type-size `{:#?}` adt general variants def {}",
                       layout.ty, adt_def.variants.len());
                let variant_infos: Vec<_> =
                    adt_def.variants.iter_enumerated().map(|(i, variant_def)| {
                        let fields: Vec<_> =
                            variant_def.fields.iter().map(|f| f.ident.name).collect();
                        build_variant_info(Some(variant_def.name),
                                           &fields,
                                           layout.for_variant(self, i))
                    })
                    .collect();
                record(adt_kind.into(), adt_packed, match layout.variants {
                    Variants::Tagged { ref tag, .. } => Some(tag.value.size(self)),
                    _ => None
                }, variant_infos);
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
        tail: Ty<'tcx>
    }
}

impl<'a, 'tcx> SizeSkeleton<'tcx> {
    pub fn compute(ty: Ty<'tcx>,
                   tcx: TyCtxt<'a, 'tcx, 'tcx>,
                   param_env: ty::ParamEnv<'tcx>)
                   -> Result<SizeSkeleton<'tcx>, LayoutError<'tcx>> {
        debug_assert!(!ty.has_infer_types());

        // First try computing a static layout.
        let err = match tcx.layout_of(param_env.and(ty)) {
            Ok(layout) => {
                return Ok(SizeSkeleton::Known(layout.size));
            }
            Err(err) => err
        };

        match ty.sty {
            ty::Ref(_, pointee, _) |
            ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                let non_zero = !ty.is_unsafe_ptr();
                let tail = tcx.struct_tail(pointee);
                match tail.sty {
                    ty::Param(_) | ty::Projection(_) => {
                        debug_assert!(tail.has_param_types() || tail.has_self_ty());
                        Ok(SizeSkeleton::Pointer {
                            non_zero,
                            tail: tcx.erase_regions(&tail)
                        })
                    }
                    _ => {
                        bug!("SizeSkeleton::compute({}): layout errored ({}), yet \
                              tail `{}` is not a type parameter or a projection",
                             ty, err, tail)
                    }
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
                    let fields = def.variants[i].fields.iter().map(|field| {
                        SizeSkeleton::compute(field.ty(tcx, substs), tcx, param_env)
                    });
                    let mut ptr = None;
                    for field in fields {
                        let field = field?;
                        match field {
                            SizeSkeleton::Known(size) => {
                                if size.bytes() > 0 {
                                    return Err(err);
                                }
                            }
                            SizeSkeleton::Pointer {..} => {
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
                            non_zero: non_zero || match tcx.layout_scalar_valid_range(def.did) {
                                (Bound::Included(start), Bound::Unbounded) => start > 0,
                                (Bound::Included(start), Bound::Included(end)) =>
                                    0 < start && start < end,
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
                    (Some(SizeSkeleton::Pointer { non_zero: true, tail }), None) |
                    (None, Some(SizeSkeleton::Pointer { non_zero: true, tail })) => {
                        Ok(SizeSkeleton::Pointer {
                            non_zero: false,
                            tail,
                        })
                    }
                    _ => Err(err)
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

            _ => Err(err)
        }
    }

    pub fn same_size(self, other: SizeSkeleton<'_>) -> bool {
        match (self, other) {
            (SizeSkeleton::Known(a), SizeSkeleton::Known(b)) => a == b,
            (SizeSkeleton::Pointer { tail: a, .. },
             SizeSkeleton::Pointer { tail: b, .. }) => a == b,
            _ => false
        }
    }
}

pub trait HasTyCtxt<'tcx>: HasDataLayout {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx>;
}

impl<'a, 'gcx, 'tcx> HasDataLayout for TyCtxt<'a, 'gcx, 'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.data_layout
    }
}

impl<'a, 'gcx, 'tcx> HasTyCtxt<'gcx> for TyCtxt<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'gcx> {
        self.global_tcx()
    }
}

impl<'tcx, T: HasDataLayout> HasDataLayout for LayoutCx<'tcx, T> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.tcx.data_layout()
    }
}

impl<'gcx, 'tcx, T: HasTyCtxt<'gcx>> HasTyCtxt<'gcx> for LayoutCx<'tcx, T> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'gcx> {
        self.tcx.tcx()
    }
}

pub trait MaybeResult<T> {
    fn from_ok(x: T) -> Self;
    fn map_same<F: FnOnce(T) -> T>(self, f: F) -> Self;
}

impl<T> MaybeResult<T> for T {
    fn from_ok(x: T) -> Self {
        x
    }
    fn map_same<F: FnOnce(T) -> T>(self, f: F) -> Self {
        f(self)
    }
}

impl<T, E> MaybeResult<T> for Result<T, E> {
    fn from_ok(x: T) -> Self {
        Ok(x)
    }
    fn map_same<F: FnOnce(T) -> T>(self, f: F) -> Self {
        self.map(f)
    }
}

pub type TyLayout<'tcx> = ::rustc_target::abi::TyLayout<'tcx, Ty<'tcx>>;

impl<'a, 'tcx> LayoutOf for LayoutCx<'tcx, TyCtxt<'a, 'tcx, 'tcx>> {
    type Ty = Ty<'tcx>;
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    /// Computes the layout of a type. Note that this implicitly
    /// executes in "reveal all" mode.
    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        let param_env = self.param_env.with_reveal_all();
        let ty = self.tcx.normalize_erasing_regions(param_env, ty);
        let details = self.tcx.layout_raw(param_env.and(ty))?;
        let layout = TyLayout {
            ty,
            details
        };

        // N.B., this recording is normally disabled; when enabled, it
        // can however trigger recursive invocations of `layout_of`.
        // Therefore, we execute it *after* the main query has
        // completed, to avoid problems around recursive structures
        // and the like. (Admittedly, I wasn't able to reproduce a problem
        // here, but it seems like the right thing to do. -nmatsakis)
        self.record_layout_for_printing(layout);

        Ok(layout)
    }
}

impl<'a, 'tcx> LayoutOf for LayoutCx<'tcx, ty::query::TyCtxtAt<'a, 'tcx, 'tcx>> {
    type Ty = Ty<'tcx>;
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    /// Computes the layout of a type. Note that this implicitly
    /// executes in "reveal all" mode.
    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        let param_env = self.param_env.with_reveal_all();
        let ty = self.tcx.normalize_erasing_regions(param_env, ty);
        let details = self.tcx.layout_raw(param_env.and(ty))?;
        let layout = TyLayout {
            ty,
            details
        };

        // N.B., this recording is normally disabled; when enabled, it
        // can however trigger recursive invocations of `layout_of`.
        // Therefore, we execute it *after* the main query has
        // completed, to avoid problems around recursive structures
        // and the like. (Admittedly, I wasn't able to reproduce a problem
        // here, but it seems like the right thing to do. -nmatsakis)
        let cx = LayoutCx {
            tcx: *self.tcx,
            param_env: self.param_env
        };
        cx.record_layout_for_printing(layout);

        Ok(layout)
    }
}

// Helper (inherent) `layout_of` methods to avoid pushing `LayoutCx` to users.
impl TyCtxt<'a, 'tcx, '_> {
    /// Computes the layout of a type. Note that this implicitly
    /// executes in "reveal all" mode.
    #[inline]
    pub fn layout_of(self, param_env_and_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                     -> Result<TyLayout<'tcx>, LayoutError<'tcx>> {
        let cx = LayoutCx {
            tcx: self.global_tcx(),
            param_env: param_env_and_ty.param_env
        };
        cx.layout_of(param_env_and_ty.value)
    }
}

impl ty::query::TyCtxtAt<'a, 'tcx, '_> {
    /// Computes the layout of a type. Note that this implicitly
    /// executes in "reveal all" mode.
    #[inline]
    pub fn layout_of(self, param_env_and_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                     -> Result<TyLayout<'tcx>, LayoutError<'tcx>> {
        let cx = LayoutCx {
            tcx: self.global_tcx().at(self.span),
            param_env: param_env_and_ty.param_env
        };
        cx.layout_of(param_env_and_ty.value)
    }
}

impl<'a, 'tcx, C> TyLayoutMethods<'tcx, C> for Ty<'tcx>
    where C: LayoutOf<Ty = Ty<'tcx>> + HasTyCtxt<'tcx>,
          C::TyLayout: MaybeResult<TyLayout<'tcx>>
{
    fn for_variant(this: TyLayout<'tcx>, cx: &C, variant_index: VariantIdx) -> TyLayout<'tcx> {
        let details = match this.variants {
            Variants::Single { index } if index == variant_index => this.details,

            Variants::Single { index } => {
                // Deny calling for_variant more than once for non-Single enums.
                cx.layout_of(this.ty).map_same(|layout| {
                    assert_eq!(layout.variants, Variants::Single { index });
                    layout
                });

                let fields = match this.ty.sty {
                    ty::Adt(def, _) => def.variants[variant_index].fields.len(),
                    _ => bug!()
                };
                let tcx = cx.tcx();
                tcx.intern_layout(LayoutDetails {
                    variants: Variants::Single { index: variant_index },
                    fields: FieldPlacement::Union(fields),
                    abi: Abi::Uninhabited,
                    align: tcx.data_layout.i8_align,
                    size: Size::ZERO
                })
            }

            Variants::NicheFilling { ref variants, .. } |
            Variants::Tagged { ref variants, .. } => {
                &variants[variant_index]
            }
        };

        assert_eq!(details.variants, Variants::Single { index: variant_index });

        TyLayout {
            ty: this.ty,
            details
        }
    }

    fn field(this: TyLayout<'tcx>, cx: &C, i: usize) -> C::TyLayout {
        let tcx = cx.tcx();
        cx.layout_of(match this.ty.sty {
            ty::Bool |
            ty::Char |
            ty::Int(_) |
            ty::Uint(_) |
            ty::Float(_) |
            ty::FnPtr(_) |
            ty::Never |
            ty::FnDef(..) |
            ty::GeneratorWitness(..) |
            ty::Foreign(..) |
            ty::Dynamic(..) => {
                bug!("TyLayout::field_type({:?}): not applicable", this)
            }

            // Potentially-fat pointers.
            ty::Ref(_, pointee, _) |
            ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                assert!(i < this.fields.count());

                // Reuse the fat *T type as its own thin pointer data field.
                // This provides information about e.g., DST struct pointees
                // (which may have no non-DST form), and will work as long
                // as the `Abi` or `FieldPlacement` is checked by users.
                if i == 0 {
                    let nil = tcx.mk_unit();
                    let ptr_ty = if this.ty.is_unsafe_ptr() {
                        tcx.mk_mut_ptr(nil)
                    } else {
                        tcx.mk_mut_ref(tcx.types.re_static, nil)
                    };
                    return cx.layout_of(ptr_ty).map_same(|mut ptr_layout| {
                        ptr_layout.ty = this.ty;
                        ptr_layout
                    });
                }

                match tcx.struct_tail(pointee).sty {
                    ty::Slice(_) |
                    ty::Str => tcx.types.usize,
                    ty::Dynamic(_, _) => {
                        tcx.mk_imm_ref(
                            tcx.types.re_static,
                            tcx.mk_array(tcx.types.usize, 3),
                        )
                        /* FIXME use actual fn pointers
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
                    _ => bug!("TyLayout::field_type({:?}): not applicable", this)
                }
            }

            // Arrays and slices.
            ty::Array(element, _) |
            ty::Slice(element) => element,
            ty::Str => tcx.types.u8,

            // Tuples, generators and closures.
            ty::Closure(def_id, ref substs) => {
                substs.upvar_tys(def_id, tcx).nth(i).unwrap()
            }

            ty::Generator(def_id, ref substs, _) => {
                substs.field_tys(def_id, tcx).nth(i).unwrap()
            }

            ty::Tuple(tys) => tys[i],

            // SIMD vector types.
            ty::Adt(def, ..) if def.repr.simd() => {
                this.ty.simd_type(tcx)
            }

            // ADTs.
            ty::Adt(def, substs) => {
                match this.variants {
                    Variants::Single { index } => {
                        def.variants[index].fields[i].ty(tcx, substs)
                    }

                    // Discriminant field for enums (where applicable).
                    Variants::Tagged { tag: ref discr, .. } |
                    Variants::NicheFilling { niche: ref discr, .. } => {
                        assert_eq!(i, 0);
                        let layout = LayoutDetails::scalar(cx, discr.clone());
                        return MaybeResult::from_ok(TyLayout {
                            details: tcx.intern_layout(layout),
                            ty: discr.value.to_ty(tcx)
                        });
                    }
                }
            }

            ty::Projection(_) | ty::UnnormalizedProjection(..) | ty::Bound(..) |
            ty::Placeholder(..) | ty::Opaque(..) | ty::Param(_) | ty::Infer(_) |
            ty::Error => {
                bug!("TyLayout::field_type: unexpected type `{}`", this.ty)
            }
        })
    }
}

struct Niche {
    offset: Size,
    scalar: Scalar,
    available: u128,
}

impl Niche {
    fn reserve<'a, 'tcx>(
        &self,
        cx: &LayoutCx<'tcx, TyCtxt<'a, 'tcx, 'tcx>>,
        count: u128,
    ) -> Option<(u128, Scalar)> {
        if count > self.available {
            return None;
        }
        let Scalar { value, valid_range: ref v } = self.scalar;
        let bits = value.size(cx).bits();
        assert!(bits <= 128);
        let max_value = !0u128 >> (128 - bits);
        let start = v.end().wrapping_add(1) & max_value;
        let end = v.end().wrapping_add(count) & max_value;
        Some((start, Scalar { value, valid_range: *v.start()..=end }))
    }
}

impl<'a, 'tcx> LayoutCx<'tcx, TyCtxt<'a, 'tcx, 'tcx>> {
    /// Find the offset of a niche leaf field, starting from
    /// the given type and recursing through aggregates.
    // FIXME(eddyb) traverse already optimized enums.
    fn find_niche(&self, layout: TyLayout<'tcx>) -> Result<Option<Niche>, LayoutError<'tcx>> {
        let scalar_niche = |scalar: &Scalar, offset| {
            let Scalar { value, valid_range: ref v } = *scalar;

            let bits = value.size(self).bits();
            assert!(bits <= 128);
            let max_value = !0u128 >> (128 - bits);

            // Find out how many values are outside the valid range.
            let available = if v.start() <= v.end() {
                v.start() + (max_value - v.end())
            } else {
                v.start() - v.end() - 1
            };

            // Give up if there is no niche value available.
            if available == 0 {
                return None;
            }

            Some(Niche { offset, scalar: scalar.clone(), available })
        };

        // Locals variables which live across yields are stored
        // in the generator type as fields. These may be uninitialized
        // so we don't look for niches there.
        if let ty::Generator(..) = layout.ty.sty {
            return Ok(None);
        }

        match layout.abi {
            Abi::Scalar(ref scalar) => {
                return Ok(scalar_niche(scalar, Size::ZERO));
            }
            Abi::ScalarPair(ref a, ref b) => {
                // HACK(nox): We iter on `b` and then `a` because `max_by_key`
                // returns the last maximum.
                let niche = iter::once(
                    (b, a.value.size(self).align_to(b.value.align(self).abi))
                )
                    .chain(iter::once((a, Size::ZERO)))
                    .filter_map(|(scalar, offset)| scalar_niche(scalar, offset))
                    .max_by_key(|niche| niche.available);
                return Ok(niche);
            }
            Abi::Vector { ref element, .. } => {
                return Ok(scalar_niche(element, Size::ZERO));
            }
            _ => {}
        }

        // Perhaps one of the fields is non-zero, let's recurse and find out.
        if let FieldPlacement::Union(_) = layout.fields {
            // Only Rust enums have safe-to-inspect fields
            // (a discriminant), other unions are unsafe.
            if let Variants::Single { .. } = layout.variants {
                return Ok(None);
            }
        }
        if let FieldPlacement::Array { .. } = layout.fields {
            if layout.fields.count() > 0 {
                return self.find_niche(layout.field(self, 0)?);
            } else {
                return Ok(None);
            }
        }
        let mut niche = None;
        let mut available = 0;
        for i in 0..layout.fields.count() {
            if let Some(mut c) = self.find_niche(layout.field(self, i)?)? {
                if c.available > available {
                    available = c.available;
                    c.offset += layout.fields.offset(i);
                    niche = Some(c);
                }
            }
        }
        Ok(niche)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Variants {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use ty::layout::Variants::*;
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Single { index } => {
                index.hash_stable(hcx, hasher);
            }
            Tagged {
                ref tag,
                ref variants,
            } => {
                tag.hash_stable(hcx, hasher);
                variants.hash_stable(hcx, hasher);
            }
            NicheFilling {
                dataful_variant,
                ref niche_variants,
                ref niche,
                niche_start,
                ref variants,
            } => {
                dataful_variant.hash_stable(hcx, hasher);
                niche_variants.start().hash_stable(hcx, hasher);
                niche_variants.end().hash_stable(hcx, hasher);
                niche.hash_stable(hcx, hasher);
                niche_start.hash_stable(hcx, hasher);
                variants.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for FieldPlacement {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use ty::layout::FieldPlacement::*;
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Union(count) => {
                count.hash_stable(hcx, hasher);
            }
            Array { count, stride } => {
                count.hash_stable(hcx, hasher);
                stride.hash_stable(hcx, hasher);
            }
            Arbitrary { ref offsets, ref memory_index } => {
                offsets.hash_stable(hcx, hasher);
                memory_index.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for VariantIdx {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'a>,
        hasher: &mut StableHasher<W>,
    ) {
        self.as_u32().hash_stable(hcx, hasher)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Abi {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use ty::layout::Abi::*;
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Uninhabited => {}
            Scalar(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ScalarPair(ref a, ref b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            Vector { ref element, count } => {
                element.hash_stable(hcx, hasher);
                count.hash_stable(hcx, hasher);
            }
            Aggregate { sized } => {
                sized.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Scalar {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let Scalar { value, ref valid_range } = *self;
        value.hash_stable(hcx, hasher);
        valid_range.start().hash_stable(hcx, hasher);
        valid_range.end().hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ::ty::layout::LayoutDetails {
    variants,
    fields,
    abi,
    size,
    align
});

impl_stable_hash_for!(enum ::ty::layout::Integer {
    I8,
    I16,
    I32,
    I64,
    I128
});

impl_stable_hash_for!(enum ::ty::layout::Primitive {
    Int(integer, signed),
    Float(fty),
    Pointer
});

impl_stable_hash_for!(struct ::ty::layout::AbiAndPrefAlign {
    abi,
    pref
});

impl<'gcx> HashStable<StableHashingContext<'gcx>> for Align {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        self.bytes().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for Size {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        self.bytes().hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for LayoutError<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use ty::layout::LayoutError::*;
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Unknown(t) |
            SizeOverflow(t) => t.hash_stable(hcx, hasher)
        }
    }
}
