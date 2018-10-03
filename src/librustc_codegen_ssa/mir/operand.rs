// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::interpret::{ConstValue, ConstEvalErr};
use rustc::mir;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, LayoutOf, TyLayout, HasTyCtxt};
use rustc_data_structures::sync::Lrc;

use base;
use MemFlags;
use glue;

use interfaces::*;

use std::fmt;

use super::{FunctionCx, LocalRef};
use super::place::PlaceRef;

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone, Debug)]
pub enum OperandValue<V> {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    /// The second value, if any, is the extra data (vtable or length)
    /// which indicates that it refers to an unsized rvalue.
    Ref(V, Option<V>, Align),
    /// A single LLVM value.
    Immediate(V),
    /// A pair of immediate LLVM values. Used by fat pointers too.
    Pair(V, V)
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
    // The value.
    pub val: OperandValue<V>,

    // The layout of value, based on its Rust type.
    pub layout: TyLayout<'tcx>,
}

impl<V : CodegenObject> fmt::Debug for OperandRef<'tcx, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, 'll: 'a, 'tcx: 'll, V : 'll + CodegenObject> OperandRef<'tcx, V> {
    pub fn new_zst<Cx: CodegenMethods<'ll, 'tcx>>(
        cx: &Cx,
        layout: TyLayout<'tcx>
    ) -> OperandRef<'tcx, V> where Cx : Backend<'ll, Value = V> {
        assert!(layout.is_zst());
        OperandRef {
            val: OperandValue::Immediate(cx.const_undef(cx.immediate_backend_type(&layout))),
            layout
        }
    }

    pub fn from_const<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        bx: &Bx,
        val: &'tcx ty::Const<'tcx>
    ) -> Result<OperandRef<'tcx, V>, Lrc<ConstEvalErr<'tcx>>> where
        Bx::CodegenCx : Backend<'ll, Value = V>,
        &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        let layout = bx.cx().layout_of(val.ty);

        if layout.is_zst() {
            return Ok(OperandRef::new_zst(bx.cx(), layout));
        }

        let val = match val.val {
            ConstValue::Unevaluated(..) => bug!(),
            ConstValue::Scalar(x) => {
                let scalar = match layout.abi {
                    layout::Abi::Scalar(ref x) => x,
                    _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                };
                let llval = bx.cx().scalar_to_backend(
                    x,
                    scalar,
                    bx.cx().immediate_backend_type(&layout),
                );
                OperandValue::Immediate(llval)
            },
            ConstValue::ScalarPair(a, b) => {
                let (a_scalar, b_scalar) = match layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (a, b),
                    _ => bug!("from_const: invalid ScalarPair layout: {:#?}", layout)
                };
                let a_llval = bx.cx().scalar_to_backend(
                    a,
                    a_scalar,
                    bx.cx().scalar_pair_element_backend_type(&layout, 0, true),
                );
                let b_layout = bx.cx().scalar_pair_element_backend_type(&layout, 1, true);
                let b_llval = bx.cx().scalar_to_backend(
                    b,
                    b_scalar,
                    b_layout,
                );
                OperandValue::Pair(a_llval, b_llval)
            },
            ConstValue::ByRef(_, alloc, offset) => {
                return Ok(bx.load_ref(&bx.cx().from_const_alloc(layout, alloc, offset)));
            },
        };

        Ok(OperandRef {
            val,
            layout
        })
    }
}

impl<'a, 'll: 'a, 'tcx: 'll, V : 'll + CodegenObject> OperandRef<'tcx, V> {
    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> V {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self)
        }
    }

    pub fn deref<Cx: 'a + CodegenMethods<'ll, 'tcx>>(
        self,
        cx: &'a Cx
    ) -> PlaceRef<'tcx, V> where
        Cx: Backend<'ll, Value=V>,
        &'a Cx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        let projected_ty = self.layout.ty.builtin_deref(true)
            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", self)).ty;
        let (llptr, llextra) = match self.val {
            OperandValue::Immediate(llptr) => (llptr, None),
            OperandValue::Pair(llptr, llextra) => (llptr, Some(llextra)),
            OperandValue::Ref(..) => bug!("Deref of by-Ref operand {:?}", self)
        };
        let layout = cx.layout_of(projected_ty);
        PlaceRef {
            llval: llptr,
            llextra,
            layout,
            align: layout.align,
        }
    }

    /// If this operand is a `Pair`, we return an aggregate with the two values.
    /// For other cases, see `immediate`.
    pub fn immediate_or_packed_pair<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        self,
        bx: &Bx
    ) -> V where Bx::CodegenCx : Backend<'ll, Value=V> {
        if let OperandValue::Pair(a, b) = self.val {
            let llty = bx.cx().backend_type(&self.layout);
            debug!("Operand::immediate_or_packed_pair: packing {:?} into {:?}",
                   self, llty);
            // Reconstruct the immediate aggregate.
            let mut llpair = bx.cx().const_undef(llty);
            llpair = bx.insert_value(llpair, base::from_immediate(bx, a), 0);
            llpair = bx.insert_value(llpair, base::from_immediate(bx, b), 1);
            llpair
        } else {
            self.immediate()
        }
    }

    /// If the type is a pair, we return a `Pair`, otherwise, an `Immediate`.
    pub fn from_immediate_or_packed_pair<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        bx: &Bx,
        llval: <Bx::CodegenCx as Backend<'ll>>::Value,
        layout: TyLayout<'tcx>
    ) -> OperandRef<'tcx, <Bx::CodegenCx as Backend<'ll>>::Value>
        where Bx::CodegenCx : Backend<'ll, Value=V>
    {
        let val = if let layout::Abi::ScalarPair(ref a, ref b) = layout.abi {
            debug!("Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}",
                    llval, layout);

            // Deconstruct the immediate aggregate.
            let a_llval = base::to_immediate_scalar(bx, bx.extract_value(llval, 0), a);
            let b_llval = base::to_immediate_scalar(bx, bx.extract_value(llval, 1), b);
            OperandValue::Pair(a_llval, b_llval)
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout }
    }

    pub fn extract_field<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        &self, bx: &Bx,
        i: usize
    ) -> OperandRef<'tcx, <Bx::CodegenCx as Backend<'ll>>::Value> where
        Bx::CodegenCx : Backend<'ll, Value=V>,
        &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        let field = self.layout.field(bx.cx(), i);
        let offset = self.layout.fields.offset(i);

        let mut val = match (self.val, &self.layout.abi) {
            // If the field is ZST, it has no data.
            _ if field.is_zst() => {
                return OperandRef::new_zst(bx.cx(), field);
            }

            // Newtype of a scalar, scalar pair or vector.
            (OperandValue::Immediate(_), _) |
            (OperandValue::Pair(..), _) if field.size == self.layout.size => {
                assert_eq!(offset.bytes(), 0);
                self.val
            }

            // Extract a scalar component from a pair.
            (OperandValue::Pair(a_llval, b_llval), &layout::Abi::ScalarPair(ref a, ref b)) => {
                if offset.bytes() == 0 {
                    assert_eq!(field.size, a.value.size(bx.cx()));
                    OperandValue::Immediate(a_llval)
                } else {
                    assert_eq!(offset, a.value.size(bx.cx())
                        .abi_align(b.value.align(bx.cx())));
                    assert_eq!(field.size, b.value.size(bx.cx()));
                    OperandValue::Immediate(b_llval)
                }
            }

            // `#[repr(simd)]` types are also immediate.
            (OperandValue::Immediate(llval), &layout::Abi::Vector { .. }) => {
                OperandValue::Immediate(
                    bx.extract_element(llval, bx.cx().const_usize(i as u64)))
            }

            _ => bug!("OperandRef::extract_field({:?}): not applicable", self)
        };

        // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
        match val {
            OperandValue::Immediate(ref mut llval) => {
                *llval = bx.bitcast(*llval, bx.cx().immediate_backend_type(&field));
            }
            OperandValue::Pair(ref mut a, ref mut b) => {
                *a = bx.bitcast(*a, bx.cx().scalar_pair_element_backend_type(&field, 0, true));
                *b = bx.bitcast(*b, bx.cx().scalar_pair_element_backend_type(&field, 1, true));
            }
            OperandValue::Ref(..) => bug!()
        }

        OperandRef {
            val,
            layout: field
        }
    }
}

impl<'a, 'll: 'a, 'tcx: 'll, V : 'll + CodegenObject> OperandValue<V> {
    pub fn store<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        self,
        bx: &Bx,
        dest: PlaceRef<'tcx, <Bx::CodegenCx as Backend<'ll>>::Value>
    ) where Bx::CodegenCx : Backend<'ll, Value = V> {
        self.store_with_flags(bx, dest, MemFlags::empty());
    }
}

// impl OperandValue<&'ll Value> {
//
//     pub fn volatile_store(
//         self,
//         bx: &Builder<'a, 'll, 'tcx, &'ll Value>,
//         dest: PlaceRef<'tcx, &'ll Value>
//     ) {
//         self.store_with_flags(bx, dest, MemFlags::VOLATILE);
//     }
//
//     pub fn unaligned_volatile_store(
//         self,
//         bx: &Builder<'a, 'll, 'tcx, &'ll Value>,
//         dest: PlaceRef<'tcx, &'ll Value>
//     ) {
//         self.store_with_flags(bx, dest, MemFlags::VOLATILE | MemFlags::UNALIGNED);
//     }
// }
//
// impl<'a, 'll: 'a, 'tcx: 'll> OperandValue<&'ll Value> {
//     pub fn nontemporal_store(
//         self,
//         bx: &Builder<'a, 'll, 'tcx, &'ll Value>,
//         dest: PlaceRef<'tcx, &'ll Value>
//     ) {
//         self.store_with_flags(bx, dest, MemFlags::NONTEMPORAL);
//     }
// }

impl<'a, 'll: 'a, 'tcx: 'll, V : 'll + CodegenObject> OperandValue<V> {
    fn store_with_flags<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        self,
        bx: &Bx,
        dest: PlaceRef<'tcx, <Bx::CodegenCx as Backend<'ll>>::Value>,
        flags: MemFlags,
    ) where Bx::CodegenCx : Backend<'ll, Value = V> {
        debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);
        // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
        // value is through `undef`, and store itself is useless.
        if dest.layout.is_zst() {
            return;
        }
        match self {
            OperandValue::Ref(r, None, source_align) => {
                base::memcpy_ty(bx, dest.llval, r, dest.layout,
                                source_align.min(dest.align), flags)
            }
            OperandValue::Ref(_, Some(_), _) => {
                bug!("cannot directly store unsized values");
            }
            OperandValue::Immediate(s) => {
                let val = base::from_immediate(bx, s);
                bx.store_with_flags(val, dest.llval, dest.align, flags);
            }
            OperandValue::Pair(a, b) => {
                for (i, &x) in [a, b].iter().enumerate() {
                    let llptr = bx.struct_gep(dest.llval, i as u64);
                    let val = base::from_immediate(bx, x);
                    bx.store_with_flags(val, llptr, dest.align, flags);
                }
            }
        }
    }
    pub fn store_unsized<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        self,
        bx: &Bx,
        indirect_dest: PlaceRef<'tcx, V>
    ) where
        Bx::CodegenCx : Backend<'ll, Value = V>,
        &'a Bx::CodegenCx: LayoutOf<Ty=Ty<'tcx>, TyLayout=TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        debug!("OperandRef::store_unsized: operand={:?}, indirect_dest={:?}", self, indirect_dest);
        let flags = MemFlags::empty();

        // `indirect_dest` must have `*mut T` type. We extract `T` out of it.
        let unsized_ty = indirect_dest.layout.ty.builtin_deref(true)
            .unwrap_or_else(|| bug!("indirect_dest has non-pointer type: {:?}", indirect_dest)).ty;

        let (llptr, llextra) =
            if let OperandValue::Ref(llptr, Some(llextra), _) = self {
                (llptr, llextra)
            } else {
                bug!("store_unsized called with a sized value")
            };

        // FIXME: choose an appropriate alignment, or use dynamic align somehow
        let max_align = Align::from_bits(128, 128).unwrap();
        let min_align = Align::from_bits(8, 8).unwrap();

        // Allocate an appropriate region on the stack, and copy the value into it
        let (llsize, _) = glue::size_and_align_of_dst(bx, unsized_ty, Some(llextra));
        let lldst = bx.array_alloca(bx.cx().type_i8(), llsize, "unsized_tmp", max_align);
        bx.call_memcpy(lldst, llptr, llsize, min_align, flags);

        // Store the allocated region and the extra to the indirect place.
        let indirect_operand = OperandValue::Pair(lldst, llextra);
        indirect_operand.store(bx, indirect_dest);
    }
}

impl<'a, 'f, 'll: 'a + 'f, 'tcx: 'll, Cx: CodegenMethods<'ll, 'tcx>>
    FunctionCx<'a, 'f, 'll, 'tcx, Cx>
{
    fn maybe_codegen_consume_direct<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: &Bx,
        place: &mir::Place<'tcx>
    ) -> Option<OperandRef<'tcx, Cx::Value>> where
        &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        debug!("maybe_codegen_consume_direct(place={:?})", place);

        // watch out for locals that do not have an
        // alloca; they are handled somewhat differently
        if let mir::Place::Local(index) = *place {
            match self.locals[index] {
                LocalRef::Operand(Some(o)) => {
                    return Some(o);
                }
                LocalRef::Operand(None) => {
                    bug!("use of {:?} before def", place);
                }
                LocalRef::Place(..) | LocalRef::UnsizedPlace(..) => {
                    // use path below
                }
            }
        }

        // Moves out of scalar and scalar pair fields are trivial.
        if let &mir::Place::Projection(ref proj) = place {
            if let Some(o) = self.maybe_codegen_consume_direct(bx, &proj.base) {
                match proj.elem {
                    mir::ProjectionElem::Field(ref f, _) => {
                        return Some(o.extract_field(bx, f.index()));
                    }
                    mir::ProjectionElem::Index(_) |
                    mir::ProjectionElem::ConstantIndex { .. } => {
                        // ZSTs don't require any actual memory access.
                        // FIXME(eddyb) deduplicate this with the identical
                        // checks in `codegen_consume` and `extract_field`.
                        let elem = o.layout.field(bx.cx(), 0);
                        if elem.is_zst() {
                            return Some(OperandRef::new_zst(bx.cx(), elem));
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }

    pub fn codegen_consume<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: &Bx,
        place: &mir::Place<'tcx>
    ) -> OperandRef<'tcx, Cx::Value> where
        &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        debug!("codegen_consume(place={:?})", place);

        let ty = self.monomorphized_place_ty(place);
        let layout = bx.cx().layout_of(ty);

        // ZSTs don't require any actual memory access.
        if layout.is_zst() {
            return OperandRef::new_zst(bx.cx(), layout);
        }

        if let Some(o) = self.maybe_codegen_consume_direct(bx, place) {
            return o;
        }

        // for most places, to consume them we just load them
        // out from their home
        bx.load_ref(&self.codegen_place(bx, place))
    }

    pub fn codegen_operand<Bx : BuilderMethods<'a, 'll, 'tcx>>(
        &mut self,
        bx: &Bx,
        operand: &mir::Operand<'tcx>
    ) -> OperandRef<'tcx, Cx::Value> where
        Bx : BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>,
        &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        debug!("codegen_operand(operand={:?})", operand);

        match *operand {
            mir::Operand::Copy(ref place) |
            mir::Operand::Move(ref place) => {
                self.codegen_consume(bx, place)
            }

            mir::Operand::Constant(ref constant) => {
                let ty = self.monomorphize(&constant.ty);
                self.eval_mir_constant(bx, constant)
                    .and_then(|c| OperandRef::from_const(bx, c))
                    .unwrap_or_else(|err| {
                        err.report_as_error(
                            bx.tcx().at(constant.span),
                            "could not evaluate constant operand",
                        );
                        // Allow RalfJ to sleep soundly knowing that even refactorings that remove
                        // the above error (or silence it under some conditions) will not cause UB
                        let fnname = bx.cx().get_intrinsic(&("llvm.trap"));
                        bx.call(fnname, &[], None);
                        // We've errored, so we don't have to produce working code.
                        let layout = bx.cx().layout_of(ty);
                        bx.load_ref(&PlaceRef::new_sized(
                            bx.cx().const_undef(bx.cx().type_ptr_to(bx.cx().backend_type(&layout))),
                            layout,
                            layout.align,
                        ))
                    })
            }
        }
    }
}
