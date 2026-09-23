/// The raw x87 80-bit extended-precision floating-point type.
///
/// This type has a size of 80 bits, and corresponds to the store size (what a load/store touches)
/// of x87_f80. This is different to what LLVM calls the alloc size, the number of bytes used to
/// calculate field offsets or the size of an alloca.
///
/// The `x87_f80` wrapper type below corresponds to the alloc size: it applies the target-specific
/// alignment, raising the total size to 12 or 16.
///
/// NOTE: this type MUST NOT be used anywhere except in the x87_f80 wrapper below.
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), lang = "x87_f80_raw")]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
struct x87_f80_raw {}

/// The x87 80-bit extended-precision floating-point lang item.
///
/// Here we define just the lang item, the public-facing type is defined
/// in core_arch. It transparently wraps this definition.
///
/// This x87_f80 type should only be used on x86 and x86_64 targets, but is always defined
/// for the purpose of doc builds, specifically doc builds in the stdarch repo.
#[doc(hidden)]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), lang = "x87_f80")]
#[unstable(feature = "num_internals", issue = "none")]
#[allow(non_camel_case_types, dead_code)]
#[derive(Clone, Copy)]
pub struct x87_f80(x87_f80_raw);

#[unstable(feature = "num_internals", issue = "none")]
impl crate::fmt::Debug for x87_f80 {
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        f.debug_tuple("x87_f80").finish()
    }
}
