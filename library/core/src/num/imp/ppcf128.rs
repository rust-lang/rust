/// The IBM extended-precision (double-double) floating-point lang item.
///
/// Here we define just the lang item, the public-facing type is defined
/// in core_arch. It transparently wraps this definition.
///
/// The ppcf128 type should only be used on PowerPC targets, but is always defined
/// for the purpose of doc builds, specifically doc builds in the stdarch repo.
#[doc(hidden)]
#[cfg_attr(any(target_arch = "powerpc", target_arch = "powerpc64"), lang = "ppcf128")]
#[unstable(feature = "num_internals", issue = "none")]
#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ppcf128(u128);
