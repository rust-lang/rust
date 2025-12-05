//! Definitions of various DWARF-related constants.

use libc::c_uint;

/// Helper macro to let us redeclare gimli's constants as our own constants
/// with a different type, with less risk of copy-paste errors.
macro_rules! declare_constant {
    (
        $name:ident : $type:ty
    ) => {
        #[allow(non_upper_case_globals)]
        pub(crate) const $name: $type = ::gimli::constants::$name.0 as $type;

        // Assert that as-cast probably hasn't changed the value.
        const _: () = assert!($name as i128 == ::gimli::constants::$name.0 as i128);
    };
}

declare_constant!(DW_TAG_const_type: c_uint);

// DWARF languages.
declare_constant!(DW_LANG_Rust: c_uint);

// DWARF attribute type encodings.
declare_constant!(DW_ATE_boolean: c_uint);
declare_constant!(DW_ATE_float: c_uint);
declare_constant!(DW_ATE_signed: c_uint);
declare_constant!(DW_ATE_unsigned: c_uint);
declare_constant!(DW_ATE_UTF: c_uint);

// DWARF expression operators.
declare_constant!(DW_OP_deref: u64);
declare_constant!(DW_OP_plus_uconst: u64);
/// Defined by LLVM in `llvm/include/llvm/BinaryFormat/Dwarf.h`.
/// Double-checked by a static assertion in `RustWrapper.cpp`.
#[allow(non_upper_case_globals)]
pub(crate) const DW_OP_LLVM_fragment: u64 = 0x1000;
// It describes the actual value of a source variable which might not exist in registers or in memory.
#[allow(non_upper_case_globals)]
pub(crate) const DW_OP_stack_value: u64 = 0x9f;
