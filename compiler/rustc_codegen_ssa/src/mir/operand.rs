use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};

use crate::base;
use crate::common::TypeKind;
use crate::glue;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::mir;
use rustc_middle::mir::interpret::{ConstValue, Pointer, Scalar};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::Ty;
use rustc_target::abi::{Abi, Align, Size};

use std::fmt;

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone, Debug)]
pub enum OperandValue<V> {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    /// The second value, if any, is the extra data (vtable or length)
    /// which indicates that it refers to an unsized rvalue.
    ///
    /// An `OperandValue` has this variant for types which are neither
    /// `Immediate` nor `Pair`s. The backend value in this variant must be a
    /// pointer to the *non*-immediate backend type. That pointee type is the
    /// one returned by [`LayoutTypeMethods::backend_type`].
    Ref(V, Option<V>, Align),
    /// A single LLVM immediate value.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeMethods::is_backend_immediate`] returns `true`.
    /// The backend value in this variant must be the *immediate* backend type,
    /// as returned by [`LayoutTypeMethods::immediate_backend_type`].
    Immediate(V),
    /// A pair of immediate LLVM values. Used by fat pointers too.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeMethods::is_backend_scalar_pair`] returns `true`.
    /// The backend values in this variant must be the *immediate* backend types,
    /// as returned by [`LayoutTypeMethods::scalar_pair_element_backend_type`]
    /// with `immediate: true`.
    Pair(V, V),
}

/// An `OperandRef` is an "SSA" reference to a Rust value, along with
/// its type.
///
/// NOTE: unless you know a value's type exactly, you should not
/// generate LLVM opcodes acting on it and instead act via methods,
/// to avoid nasty edge cases. In particular, using `Builder::store`
/// directly is sure to cause problems -- use `OperandRef::store`
/// instead.
#[derive(Copy, Clone)]
pub struct OperandRef<'tcx, V> {
    /// The value.
    pub val: OperandValue<V>,

    /// The layout of value, based on its Rust type.
    pub layout: TyAndLayout<'tcx>,
}

impl<V: CodegenObject> fmt::Debug for OperandRef<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, 'tcx, V: CodegenObject> OperandRef<'tcx, V> {
    pub fn new_zst<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> OperandRef<'tcx, V> {
        assert!(layout.is_zst());
        OperandRef {
            val: OperandValue::Immediate(bx.const_poison(bx.immediate_backend_type(layout))),
            layout,
        }
    }

    pub fn from_const<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        val: ConstValue<'tcx>,
        ty: Ty<'tcx>,
    ) -> Self {
        let layout = bx.layout_of(ty);

        let val = match val {
            ConstValue::Scalar(x) => {
                let Abi::Scalar(scalar) = layout.abi else {
                    bug!("from_const: invalid ByVal layout: {:#?}", layout);
                };
                let llval = bx.scalar_to_backend(x, scalar, bx.immediate_backend_type(layout));
                OperandValue::Immediate(llval)
            }
            ConstValue::ZeroSized => return OperandRef::new_zst(bx, layout),
            ConstValue::Slice { data, start, end } => {
                let Abi::ScalarPair(a_scalar, _) = layout.abi else {
                    bug!("from_const: invalid ScalarPair layout: {:#?}", layout);
                };
                let a = Scalar::from_pointer(
                    Pointer::new(bx.tcx().create_memory_alloc(data), Size::from_bytes(start)),
                    &bx.tcx(),
                );
                let a_llval = bx.scalar_to_backend(
                    a,
                    a_scalar,
                    bx.scalar_pair_element_backend_type(layout, 0, true),
                );
                let b_llval = bx.const_usize((end - start) as u64);
                OperandValue::Pair(a_llval, b_llval)
            }
            ConstValue::ByRef { alloc, offset } => {
                return bx.load_operand(bx.from_const_alloc(layout, alloc, offset));
            }
        };

        OperandRef { val, layout }
    }

    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> V {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self),
        }
    }

    pub fn deref<Cx: LayoutTypeMethods<'tcx>>(self, cx: &Cx) -> PlaceRef<'tcx, V> {
        if self.layout.ty.is_box() {
            bug!("dereferencing {:?} in codegen", self.layout.ty);
        }

        let projected_ty = self
            .layout
            .ty
            .builtin_deref(true)
            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", self))
            .ty;

        let (llptr, llextra) = match self.val {
            OperandValue::Immediate(llptr) => (llptr, None),
            OperandValue::Pair(llptr, llextra) => (llptr, Some(llextra)),
            OperandValue::Ref(..) => bug!("Deref of by-Ref operand {:?}", self),
        };
        let layout = cx.layout_of(projected_ty);
        PlaceRef { llval: llptr, llextra, layout, align: layout.align.abi }
    }

    /// If this operand is a `Pair`, we return an aggregate with the two values.
    /// For other cases, see `immediate`.
    pub fn immediate_or_packed_pair<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
    ) -> V {
        if let OperandValue::Pair(a, b) = self.val {
            let llty = bx.cx().backend_type(self.layout);
            debug!("Operand::immediate_or_packed_pair: packing {:?} into {:?}", self, llty);
            // Reconstruct the immediate aggregate.
            let mut llpair = bx.cx().const_poison(llty);
            let imm_a = bx.from_immediate(a);
            let imm_b = bx.from_immediate(b);
            llpair = bx.insert_value(llpair, imm_a, 0);
            llpair = bx.insert_value(llpair, imm_b, 1);
            llpair
        } else {
            self.immediate()
        }
    }

    /// If the type is a pair, we return a `Pair`, otherwise, an `Immediate`.
    pub fn from_immediate_or_packed_pair<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        llval: V,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        let val = if let Abi::ScalarPair(a, b) = layout.abi {
            debug!("Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}", llval, layout);

            // Deconstruct the immediate aggregate.
            let a_llval = bx.extract_value(llval, 0);
            let a_llval = bx.to_immediate_scalar(a_llval, a);
            let b_llval = bx.extract_value(llval, 1);
            let b_llval = bx.to_immediate_scalar(b_llval, b);
            OperandValue::Pair(a_llval, b_llval)
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout }
    }

    pub fn extract_field<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        i: usize,
    ) -> Self {
        let field = self.layout.field(bx.cx(), i);
        let offset = self.layout.fields.offset(i);

        let mut val = match (self.val, self.layout.abi) {
            // If the field is ZST, it has no data.
            _ if field.is_zst() => {
                return OperandRef::new_zst(bx, field);
            }

            // Newtype of a scalar, scalar pair or vector.
            (OperandValue::Immediate(_) | OperandValue::Pair(..), _)
                if field.size == self.layout.size =>
            {
                assert_eq!(offset.bytes(), 0);
                self.val
            }

            // Extract a scalar component from a pair.
            (OperandValue::Pair(a_llval, b_llval), Abi::ScalarPair(a, b)) => {
                if offset.bytes() == 0 {
                    assert_eq!(field.size, a.size(bx.cx()));
                    OperandValue::Immediate(a_llval)
                } else {
                    assert_eq!(offset, a.size(bx.cx()).align_to(b.align(bx.cx()).abi));
                    assert_eq!(field.size, b.size(bx.cx()));
                    OperandValue::Immediate(b_llval)
                }
            }

            // `#[repr(simd)]` types are also immediate.
            (OperandValue::Immediate(llval), Abi::Vector { .. }) => {
                OperandValue::Immediate(bx.extract_element(llval, bx.cx().const_usize(i as u64)))
            }

            _ => bug!("OperandRef::extract_field({:?}): not applicable", self),
        };

        match (&mut val, field.abi) {
            (
                OperandValue::Immediate(llval),
                Abi::Scalar(_) | Abi::ScalarPair(..) | Abi::Vector { .. },
            ) => {
                // Bools in union fields needs to be truncated.
                *llval = bx.to_immediate(*llval, field);
                // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
                let ty = bx.cx().immediate_backend_type(field);
                if bx.type_kind(ty) == TypeKind::Pointer {
                    *llval = bx.pointercast(*llval, ty);
                }
            }
            (OperandValue::Pair(a, b), Abi::ScalarPair(a_abi, b_abi)) => {
                // Bools in union fields needs to be truncated.
                *a = bx.to_immediate_scalar(*a, a_abi);
                *b = bx.to_immediate_scalar(*b, b_abi);
                // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
                let a_ty = bx.cx().scalar_pair_element_backend_type(field, 0, true);
                let b_ty = bx.cx().scalar_pair_element_backend_type(field, 1, true);
                if bx.type_kind(a_ty) == TypeKind::Pointer {
                    *a = bx.pointercast(*a, a_ty);
                }
                if bx.type_kind(b_ty) == TypeKind::Pointer {
                    *b = bx.pointercast(*b, b_ty);
                }
            }
            // Newtype vector of array, e.g. #[repr(simd)] struct S([i32; 4]);
            (OperandValue::Immediate(llval), Abi::Aggregate { sized: true }) => {
                assert!(matches!(self.layout.abi, Abi::Vector { .. }));

                let llty = bx.cx().backend_type(self.layout);
                let llfield_ty = bx.cx().backend_type(field);

                // Can't bitcast an aggregate, so round trip through memory.
                let lltemp = bx.alloca(llfield_ty, field.align.abi);
                let llptr = bx.pointercast(lltemp, bx.cx().type_ptr_to(llty));
                bx.store(*llval, llptr, field.align.abi);
                *llval = bx.load(llfield_ty, lltemp, field.align.abi);
            }
            (OperandValue::Immediate(_), Abi::Uninhabited | Abi::Aggregate { sized: false }) => {
                bug!()
            }
            (OperandValue::Pair(..), _) => bug!(),
            (OperandValue::Ref(..), _) => bug!(),
        }

        OperandRef { val, layout: field }
    }
}

impl<'a, 'tcx, V: CodegenObject> OperandValue<V> {
    /// Returns an `OperandValue` that's generally UB to use in any way.
    ///
    /// Depending on the `layout`, returns an `Immediate` or `Pair` containing
    /// poison value(s), or a `Ref` containing a poison pointer.
    ///
    /// Supports sized types only.
    pub fn poison<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> OperandValue<V> {
        assert!(layout.is_sized());
        if bx.cx().is_backend_immediate(layout) {
            let ibty = bx.cx().immediate_backend_type(layout);
            OperandValue::Immediate(bx.const_poison(ibty))
        } else if bx.cx().is_backend_scalar_pair(layout) {
            let ibty0 = bx.cx().scalar_pair_element_backend_type(layout, 0, true);
            let ibty1 = bx.cx().scalar_pair_element_backend_type(layout, 1, true);
            OperandValue::Pair(bx.const_poison(ibty0), bx.const_poison(ibty1))
        } else {
            let bty = bx.cx().backend_type(layout);
            let ptr_bty = bx.cx().type_ptr_to(bty);
            OperandValue::Ref(bx.const_poison(ptr_bty), None, layout.align.abi)
        }
    }

    pub fn store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::empty());
    }

    pub fn volatile_store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::VOLATILE);
    }

    pub fn unaligned_volatile_store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::VOLATILE | MemFlags::UNALIGNED);
    }

    pub fn nontemporal_store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::NONTEMPORAL);
    }

    fn store_with_flags<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
        flags: MemFlags,
    ) {
        debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);
        // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
        // value is through `undef`, and store itself is useless.
        if dest.layout.is_zst() {
            return;
        }
        match self {
            OperandValue::Ref(r, None, source_align) => {
                if flags.contains(MemFlags::NONTEMPORAL) {
                    // HACK(nox): This is inefficient but there is no nontemporal memcpy.
                    let ty = bx.backend_type(dest.layout);
                    let ptr = bx.pointercast(r, bx.type_ptr_to(ty));
                    let val = bx.load(ty, ptr, source_align);
                    bx.store_with_flags(val, dest.llval, dest.align, flags);
                    return;
                }
                base::memcpy_ty(bx, dest.llval, dest.align, r, source_align, dest.layout, flags)
            }
            OperandValue::Ref(_, Some(_), _) => {
                bug!("cannot directly store unsized values");
            }
            OperandValue::Immediate(s) => {
                let val = bx.from_immediate(s);
                bx.store_with_flags(val, dest.llval, dest.align, flags);
            }
            OperandValue::Pair(a, b) => {
                let Abi::ScalarPair(a_scalar, b_scalar) = dest.layout.abi else {
                    bug!("store_with_flags: invalid ScalarPair layout: {:#?}", dest.layout);
                };
                let ty = bx.backend_type(dest.layout);
                let b_offset = a_scalar.size(bx).align_to(b_scalar.align(bx).abi);

                let llptr = bx.struct_gep(ty, dest.llval, 0);
                let val = bx.from_immediate(a);
                let align = dest.align;
                bx.store_with_flags(val, llptr, align, flags);

                let llptr = bx.struct_gep(ty, dest.llval, 1);
                let val = bx.from_immediate(b);
                let align = dest.align.restrict_for_offset(b_offset);
                bx.store_with_flags(val, llptr, align, flags);
            }
        }
    }

    pub fn store_unsized<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        indirect_dest: PlaceRef<'tcx, V>,
    ) {
        debug!("OperandRef::store_unsized: operand={:?}, indirect_dest={:?}", self, indirect_dest);
        let flags = MemFlags::empty();

        // `indirect_dest` must have `*mut T` type. We extract `T` out of it.
        let unsized_ty = indirect_dest
            .layout
            .ty
            .builtin_deref(true)
            .unwrap_or_else(|| bug!("indirect_dest has non-pointer type: {:?}", indirect_dest))
            .ty;

        let OperandValue::Ref(llptr, Some(llextra), _) = self else {
            bug!("store_unsized called with a sized value")
        };

        // FIXME: choose an appropriate alignment, or use dynamic align somehow
        let max_align = Align::from_bits(128).unwrap();
        let min_align = Align::from_bits(8).unwrap();

        // Allocate an appropriate region on the stack, and copy the value into it
        let (llsize, _) = glue::size_and_align_of_dst(bx, unsized_ty, Some(llextra));
        let lldst = bx.byte_array_alloca(llsize, max_align);
        bx.memcpy(lldst, max_align, llptr, min_align, llsize, flags);

        // Store the allocated region and the extra to the indirect place.
        let indirect_operand = OperandValue::Pair(lldst, llextra);
        indirect_operand.store(bx, indirect_dest);
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    fn maybe_codegen_consume_direct(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::PlaceRef<'tcx>,
    ) -> Option<OperandRef<'tcx, Bx::Value>> {
        debug!("maybe_codegen_consume_direct(place_ref={:?})", place_ref);

        match self.locals[place_ref.local] {
            LocalRef::Operand(mut o) => {
                // Moves out of scalar and scalar pair fields are trivial.
                for elem in place_ref.projection.iter() {
                    match elem {
                        mir::ProjectionElem::Field(ref f, _) => {
                            o = o.extract_field(bx, f.index());
                        }
                        mir::ProjectionElem::Index(_)
                        | mir::ProjectionElem::ConstantIndex { .. } => {
                            // ZSTs don't require any actual memory access.
                            // FIXME(eddyb) deduplicate this with the identical
                            // checks in `codegen_consume` and `extract_field`.
                            let elem = o.layout.field(bx.cx(), 0);
                            if elem.is_zst() {
                                o = OperandRef::new_zst(bx, elem);
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }

                Some(o)
            }
            LocalRef::PendingOperand => {
                bug!("use of {:?} before def", place_ref);
            }
            LocalRef::Place(..) | LocalRef::UnsizedPlace(..) => {
                // watch out for locals that do not have an
                // alloca; they are handled somewhat differently
                None
            }
        }
    }

    pub fn codegen_consume(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::PlaceRef<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        debug!("codegen_consume(place_ref={:?})", place_ref);

        let ty = self.monomorphized_place_ty(place_ref);
        let layout = bx.cx().layout_of(ty);

        // ZSTs don't require any actual memory access.
        if layout.is_zst() {
            return OperandRef::new_zst(bx, layout);
        }

        if let Some(o) = self.maybe_codegen_consume_direct(bx, place_ref) {
            return o;
        }

        // for most places, to consume them we just load them
        // out from their home
        let place = self.codegen_place(bx, place_ref);
        bx.load_operand(place)
    }

    pub fn codegen_operand(
        &mut self,
        bx: &mut Bx,
        operand: &mir::Operand<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        debug!("codegen_operand(operand={:?})", operand);

        match *operand {
            mir::Operand::Copy(ref place) | mir::Operand::Move(ref place) => {
                self.codegen_consume(bx, place.as_ref())
            }

            mir::Operand::Constant(ref constant) => {
                // This cannot fail because we checked all required_consts in advance.
                self.eval_mir_constant_to_operand(bx, constant).unwrap_or_else(|_err| {
                    span_bug!(constant.span, "erroneous constant not captured by required_consts")
                })
            }
        }
    }
}
