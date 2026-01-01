use rustc_abi as abi;
use rustc_middle::mir::interpret::{ConstAllocation, GlobalAlloc, Scalar};
use rustc_middle::ty::Instance;

use super::BackendTypes;

pub trait ConstCodegenMethods<'tcx>: BackendTypes {
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

    /// Turn a `GlobalAlloc` into a backend global, return the value and instance that is used to
    /// generate the symbol name, if any.
    ///
    /// If the `GlobalAlloc` should not be mapped to a global, but absolute address should be used,
    /// an integer is returned as `Err` instead.
    ///
    /// If the caller needs to guarantee a symbol name, it can provide a name hint. The name will be
    /// used to generate a new symbol if there isn't one already (i.e. the case of fn/static).
    fn alloc_to_backend(
        &self,
        global_alloc: GlobalAlloc<'tcx>,
        name_hint: Option<Instance<'tcx>>,
    ) -> Result<(Self::Value, Option<Instance<'tcx>>), u64>;
    fn scalar_to_backend(&self, cv: Scalar, layout: abi::Scalar, llty: Self::Type) -> Self::Value;

    fn const_ptr_byte_offset(&self, val: Self::Value, offset: abi::Size) -> Self::Value;
}
