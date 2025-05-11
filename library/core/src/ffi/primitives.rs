//! Defines primitive types that match C's type definitions for FFI compatibility.
//!
//! This module is intentionally standalone to facilitate parsing when retrieving
//! core C types.

macro_rules! type_alias {
    {
      $Docfile:tt, $Alias:ident = $Real:ty;
      $( $Cfg:tt )*
    } => {
        #[doc = include_str!($Docfile)]
        $( $Cfg )*
        #[stable(feature = "core_ffi_c", since = "1.64.0")]
        pub type $Alias = $Real;
    }
}

type_alias! { "c_char.md", c_char = c_char_definition::c_char; #[doc(cfg(all()))] }

type_alias! { "c_schar.md", c_schar = i8; }
type_alias! { "c_uchar.md", c_uchar = u8; }
type_alias! { "c_short.md", c_short = i16; }
type_alias! { "c_ushort.md", c_ushort = u16; }

type_alias! { "c_int.md", c_int = c_int_definition::c_int; #[doc(cfg(all()))] }
type_alias! { "c_uint.md", c_uint = c_int_definition::c_uint; #[doc(cfg(all()))] }

type_alias! { "c_long.md", c_long = c_long_definition::c_long; #[doc(cfg(all()))] }
type_alias! { "c_ulong.md", c_ulong = c_long_definition::c_ulong; #[doc(cfg(all()))] }

type_alias! { "c_longlong.md", c_longlong = i64; }
type_alias! { "c_ulonglong.md", c_ulonglong = u64; }

type_alias! { "c_float.md", c_float = f32; }
type_alias! { "c_double.md", c_double = f64; }

mod c_char_definition {
    crate::cfg_match! {
        // These are the targets on which c_char is unsigned. Usually the
        // signedness is the same for all target_os values on a given architecture
        // but there are some exceptions (see isSignedCharDefault() in clang).
        // aarch64:
        //   Section 10 "Arm C and C++ language mappings" in Procedure Call Standard for the Arm®
        //   64-bit Architecture (AArch64) says C/C++ char is unsigned byte.
        //   https://github.com/ARM-software/abi-aa/blob/2024Q3/aapcs64/aapcs64.rst#arm-c-and-c-language-mappings
        // arm:
        //   Section 8 "Arm C and C++ Language Mappings" in Procedure Call Standard for the Arm®
        //   Architecture says C/C++ char is unsigned byte.
        //   https://github.com/ARM-software/abi-aa/blob/2024Q3/aapcs32/aapcs32.rst#arm-c-and-c-language-mappings
        // csky:
        //   Section 2.1.2 "Primary Data Type" in C-SKY V2 CPU Applications Binary Interface
        //   Standards Manual says ANSI C char is unsigned byte.
        //   https://github.com/c-sky/csky-doc/blob/9f7121f7d40970ba5cc0f15716da033db2bb9d07/C-SKY_V2_CPU_Applications_Binary_Interface_Standards_Manual.pdf
        //   Note: this doesn't seem to match Clang's default (https://github.com/rust-lang/rust/issues/129945).
        // hexagon:
        //   Section 3.1 "Basic data type" in Qualcomm Hexagon™ Application
        //   Binary Interface User Guide says "By default, the `char` data type is unsigned."
        //   https://docs.qualcomm.com/bundle/publicresource/80-N2040-23_REV_K_Qualcomm_Hexagon_Application_Binary_Interface_User_Guide.pdf
        // msp430:
        //   Section 2.1 "Basic Types" in MSP430 Embedded Application Binary
        //   Interface says "The char type is unsigned by default".
        //   https://www.ti.com/lit/an/slaa534a/slaa534a.pdf
        // powerpc/powerpc64:
        //   - PPC32 SysV: "Table 3-1 Scalar Types" in System V Application Binary Interface PowerPC
        //     Processor Supplement says ANSI C char is unsigned byte
        //     https://refspecs.linuxfoundation.org/elf/elfspec_ppc.pdf
        //   - PPC64 ELFv1: Section 3.1.4 "Fundamental Types" in 64-bit PowerPC ELF Application
        //     Binary Interface Supplement 1.9 says ANSI C is unsigned byte
        //     https://refspecs.linuxfoundation.org/ELF/ppc64/PPC-elf64abi.html#FUND-TYPE
        //   - PPC64 ELFv2: Section 2.1.2.2 "Fundamental Types" in 64-Bit ELF V2 ABI Specification
        //     says char is unsigned byte
        //     https://openpowerfoundation.org/specifications/64bitelfabi/
        //   - AIX: XL C for AIX Language Reference says "By default, char behaves like an unsigned char."
        //     https://www.ibm.com/docs/en/xl-c-aix/13.1.3?topic=specifiers-character-types
        // riscv32/riscv64:
        //   C/C++ type representations section in RISC-V Calling Conventions
        //   page in RISC-V ELF psABI Document says "char is unsigned."
        //   https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/draft-20240829-13bfa9f54634cb60d86b9b333e109f077805b4b3/riscv-cc.adoc#cc-type-representations
        // s390x:
        //   - ELF: "Table 1.1.: Scalar types" in ELF Application Binary Interface s390x Supplement
        //     Version 1.6.1 categorize ISO C char in unsigned integer
        //     https://github.com/IBM/s390x-abi/releases/tag/v1.6.1
        //   - z/OS: XL C/C++ Language Reference says: "By default, char behaves like an unsigned char."
        //     https://www.ibm.com/docs/en/zos/3.1.0?topic=specifiers-character-types
        // xtensa:
        //   Section 2.17.1 "Data Types and Alignment" of Xtensa LX Microprocessor Overview handbook
        //   says "`char` type is unsigned by default".
        //   https://loboris.eu/ESP32/Xtensa_lx%20Overview%20handbook.pdf
        //
        // On the following operating systems, c_char is signed by default, regardless of architecture.
        // Darwin (macOS, iOS, etc.):
        //   Apple targets' c_char is signed by default even on arm
        //   https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Handle-data-types-and-data-alignment-properly
        // Windows:
        //   Windows MSVC C++ Language Reference says "Microsoft-specific: Variables of type char
        //   are promoted to int as if from type signed char by default, unless the /J compilation
        //   option is used."
        //   https://learn.microsoft.com/en-us/cpp/cpp/fundamental-types-cpp?view=msvc-170#character-types
        // Vita:
        //   Chars are signed by default on the Vita, and VITASDK follows that convention.
        //   https://github.com/vitasdk/buildscripts/blob/09c533b771591ecde88864b6acad28ffb688dbd4/patches/gcc/0001-gcc-10.patch#L33-L34
        //
        // L4Re:
        //   The kernel builds with -funsigned-char on all targets (but userspace follows the
        //   architecture defaults). As we only have a target for userspace apps so there are no
        //   special cases for L4Re below.
        //   https://github.com/rust-lang/rust/pull/132975#issuecomment-2484645240
        all(
            not(windows),
            not(target_vendor = "apple"),
            not(target_os = "vita"),
            any(
                target_arch = "aarch64",
                target_arch = "arm",
                target_arch = "csky",
                target_arch = "hexagon",
                target_arch = "msp430",
                target_arch = "powerpc",
                target_arch = "powerpc64",
                target_arch = "riscv32",
                target_arch = "riscv64",
                target_arch = "s390x",
                target_arch = "xtensa",
            )
        ) => {
            pub(super) type c_char = u8;
        }
        // On every other target, c_char is signed.
        _ => {
            pub(super) type c_char = i8;
        }
    }
}

mod c_long_definition {
    crate::cfg_match! {
        any(
            all(target_pointer_width = "64", not(windows)),
            // wasm32 Linux ABI uses 64-bit long
            all(target_arch = "wasm32", target_os = "linux")
        ) => {
            pub(super) type c_long = i64;
            pub(super) type c_ulong = u64;
        }
        _ => {
            // The minimal size of `long` in the C standard is 32 bits
            pub(super) type c_long = i32;
            pub(super) type c_ulong = u32;
        }
    }
}

/// Equivalent to C's `size_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`usize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_size_t = usize;

/// Equivalent to C's `ptrdiff_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_ptrdiff_t = isize;

/// Equivalent to C's `ssize_t` (on POSIX) or `SSIZE_T` (on Windows) type.
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_ssize_t = isize;

mod c_int_definition {
    crate::cfg_match! {
        any(target_arch = "avr", target_arch = "msp430") => {
            pub(super) type c_int = i16;
            pub(super) type c_uint = u16;
        }
        _ => {
            pub(super) type c_int = i32;
            pub(super) type c_uint = u32;
        }
    }
}
