use rustc_abi::{self as abi, Align, HasDataLayout, Primitive};
use rustc_ast::Mutability;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hashes::Hash128;
use rustc_middle::mir::interpret::{ConstAllocation, GlobalAlloc, Scalar};
use rustc_middle::ty::layout::HasTyCtxt;

use super::BaseTypeCodegenMethods;
use crate::traits::{MiscCodegenMethods, StaticCodegenMethods};

pub trait ConstCodegenMethods<'tcx>:
    BaseTypeCodegenMethods
    + HasDataLayout
    + HasTyCtxt<'tcx>
    + MiscCodegenMethods<'tcx>
    + StaticCodegenMethods
    + Sized
{
    // Constant constructors
    fn const_null(&self, t: Self::Type) -> Self::Value;
    /// Generate an uninitialized value (matching uninitialized memory in MIR).
    /// Whether memory is initialized or not is tracked byte-for-byte.
    fn const_undef(&self, t: Self::Type) -> Self::Value;
    /// Generate a fake value. Poison always affects the entire value, even if just a single byte is
    /// poison. This can only be used in codepaths that are already UB, i.e., UB-free Rust code
    /// (including code that e.g. copies uninit memory with `MaybeUninit`) can never encounter a
    /// poison value.
    fn const_poison(&self, t: Self::Type) -> Self::Value;

    fn const_bool(&self, val: bool) -> Self::Value;

    fn const_i8(&self, i: i8) -> Self::Value;
    fn const_i16(&self, i: i16) -> Self::Value;
    fn const_i32(&self, i: i32) -> Self::Value;
    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value;
    fn const_u8(&self, i: u8) -> Self::Value;
    fn const_u32(&self, i: u32) -> Self::Value;
    fn const_u64(&self, i: u64) -> Self::Value;
    fn const_u128(&self, i: u128) -> Self::Value;
    fn const_usize(&self, i: u64) -> Self::Value;
    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value;
    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value;
    fn const_real(&self, t: Self::Type, val: f64) -> Self::Value;

    fn const_str(&self, s: &str) -> (Self::Value, Self::Value);
    fn const_struct(&self, elts: &[Self::Value], packed: bool) -> Self::Value;
    fn const_vector(&self, elts: &[Self::Value]) -> Self::Value;

    fn const_to_opt_uint(&self, v: Self::Value) -> Option<u64>;
    fn const_to_opt_u128(&self, v: Self::Value, sign_ext: bool) -> Option<u128>;

    fn const_data_from_alloc(&self, alloc: ConstAllocation<'_>) -> Self::Value;

    fn const_bitcast(&self, val: Self::Value, ty: Self::Type) -> Self::Value;
    fn const_pointercast(&self, val: Self::Value, ty: Self::Type) -> Self::Value;
    fn const_int_to_ptr(&self, val: Self::Value, ty: Self::Type) -> Self::Value;
    fn const_ptr_to_int(&self, val: Self::Value, ty: Self::Type) -> Self::Value;
    /// Create a global constant.
    ///
    /// The returned global variable is a pointer in the default address space for globals.
    fn static_addr_of_const(
        &self,
        cv: Self::Value,
        align: Align,
        kind: Option<&str>,
    ) -> Self::Value;

    /// Same as `static_addr_of_const`, but does not mark the static as immutable
    fn static_addr_of_mut(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;

    fn scalar_to_backend(&self, cv: Scalar, layout: abi::Scalar, ty: Self::Type) -> Self::Value {
        let bitsize = if layout.is_bool() { 1 } else { layout.size(self).bits() };
        match cv {
            Scalar::Int(int) => {
                let data = int.to_bits(layout.size(self));
                let val = self.const_uint_big(self.type_ix(bitsize), data);
                if matches!(layout.primitive(), Primitive::Pointer(_)) {
                    self.const_int_to_ptr(val, ty)
                } else {
                    self.const_bitcast(val, ty)
                }
            }
            Scalar::Ptr(ptr, _size) => {
                let (prov, offset) = ptr.prov_and_relative_offset();
                let global_alloc = self.tcx().global_alloc(prov.alloc_id());
                let base_addr = match global_alloc {
                    GlobalAlloc::Memory(alloc) => {
                        // For ZSTs directly codegen an aligned pointer.
                        // This avoids generating a zero-sized constant value and actually needing a
                        // real address at runtime.
                        if alloc.inner().len() == 0 {
                            assert_eq!(offset.bytes(), 0);
                            let val = self.const_usize(alloc.inner().align.bytes());
                            return if matches!(layout.primitive(), Primitive::Pointer(_)) {
                                self.const_int_to_ptr(val, ty)
                            } else {
                                self.const_bitcast(val, ty)
                            };
                        } else {
                            let init = self.const_data_from_alloc(alloc);
                            let alloc = alloc.inner();
                            let value = match alloc.mutability {
                                Mutability::Mut => self.static_addr_of_mut(init, alloc.align, None),
                                _ => self.static_addr_of_const(init, alloc.align, None),
                            };
                            if !self.tcx().sess.fewer_names()
                                && self.get_value_name(value).is_empty()
                            {
                                let hash = self.tcx().with_stable_hashing_context(|mut hcx| {
                                    let mut hasher = StableHasher::new();
                                    alloc.hash_stable(&mut hcx, &mut hasher);
                                    hasher.finish::<Hash128>()
                                });
                                self.set_value_name(value, format!("alloc_{hash:032x}").as_bytes());
                            }
                            value
                        }
                    }
                    GlobalAlloc::Function { instance, .. } => self.get_fn_addr(instance),
                    GlobalAlloc::VTable(ty, dyn_ty) => {
                        let alloc = self
                            .tcx()
                            .global_alloc(self.tcx().vtable_allocation((
                                ty,
                                dyn_ty.principal().map(|principal| {
                                    self.tcx().instantiate_bound_regions_with_erased(principal)
                                }),
                            )))
                            .unwrap_memory();
                        let init = self.const_data_from_alloc(alloc);
                        let value = self.static_addr_of_const(init, alloc.inner().align, None);
                        value
                    }
                    GlobalAlloc::Static(def_id) => {
                        assert!(self.tcx().is_static(def_id));
                        assert!(!self.tcx().is_thread_local_static(def_id));
                        self.get_static(def_id)
                    }
                    GlobalAlloc::TypeId { .. } => {
                        // Drop the provenance, the offset contains the bytes of the hash
                        let val = self.const_usize(offset.bytes());
                        // This is still a variable of pointer type, even though we only use the provenance
                        // of that pointer in CTFE and Miri. But to make the backend's type system happy,
                        // we need an int-to-ptr cast here (it doesn't matter at all which provenance that picks).
                        return self.const_int_to_ptr(val, ty);
                    }
                };
                let base_addr_space = global_alloc.address_space(self);

                // Cast to the required address space if necessary
                let val = self.const_pointercast(base_addr, self.type_ptr_ext(base_addr_space));
                let val = self.const_ptr_byte_offset(val, offset);

                if !matches!(layout.primitive(), Primitive::Pointer(_)) {
                    self.const_ptr_to_int(val, ty)
                } else {
                    self.const_bitcast(val, ty)
                }
            }
        }
    }

    fn const_ptr_byte_offset(&self, val: Self::Value, offset: abi::Size) -> Self::Value;
}
