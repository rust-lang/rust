use rustc_middle::mir::interpret::{ConstAllocation, Scalar};
use rustc_target::abi;

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

    fn const_data_from_alloc(&self, alloc: ConstAllocation<'tcx>) -> Self::Value;

    fn scalar_to_backend(&self, cv: Scalar, layout: abi::Scalar, llty: Self::Type) -> Self::Value;

    fn const_ptr_byte_offset(&self, val: Self::Value, offset: abi::Size) -> Self::Value;
}
