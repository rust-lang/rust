//! AArch64 Memory tagging intrinsics
//!
//! [ACLE documentation](https://arm-software.github.io/acle/main/acle.html#markdown-toc-mte-intrinsics)

unsafe extern "unadjusted" {
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.irg"
    )]
    fn irg_(ptr: *const (), exclude: i64) -> *const ();
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.gmi"
    )]
    fn gmi_(ptr: *const (), exclude: i64) -> i64;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.ldg"
    )]
    fn ldg_(ptr: *const (), tag_ptr: *const ()) -> *const ();
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.stg"
    )]
    fn stg_(tagged_ptr: *const (), addr_to_tag: *const ());
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.addg"
    )]
    fn addg_(ptr: *const (), value: i64) -> *const ();
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.subp"
    )]
    fn subp_(ptr_a: *const (), ptr_b: *const ()) -> i64;
}

/// Return a pointer containing a randomly generated logical address tag.
///
/// `src`: A pointer containing an address.
/// `mask`: A mask where each of the lower 16 bits specifies logical
///         tags which must be excluded from consideration. Zero excludes no
///         tags.
///
/// The returned pointer contains a copy of the `src` address, but with a
/// randomly generated logical tag, excluding any specified by `mask`.
///
/// SAFETY: The pointer provided by this intrinsic will be invalid until the memory
/// has been appropriately tagged with `__arm_mte_set_tag`. If using that intrinsic
/// on the provided pointer is itself invalid, then it will be permanently invalid
/// and Undefined Behavior to dereference it.
#[inline]
#[target_feature(enable = "mte")]
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub unsafe fn __arm_mte_create_random_tag<T>(src: *const T, mask: u64) -> *const T {
    irg_(src as *const (), mask as i64) as *const T
}

/// Return a pointer with the logical address tag offset by a value.
///
/// `src`: A pointer containing an address and a logical tag.
/// `OFFSET`: A compile-time constant value in the range [0, 15].
///
/// Adds offset to the logical address tag in `src`, wrapping if the result is
/// outside of the valid 16 tags.
///
/// SAFETY: See `__arm_mte_create_random_tag`.
#[inline]
#[target_feature(enable = "mte")]
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub unsafe fn __arm_mte_increment_tag<const OFFSET: i64, T>(src: *const T) -> *const T {
    addg_(src as *const (), OFFSET) as *const T
}

/// Add a logical tag to the set of excluded logical tags.
///
/// `src`: A pointer containing an address and a logical tag.
/// `excluded`: A mask where the lower 16 bits each specify currently-excluded
///             logical tags.
///
/// Adds the logical tag stored in `src` to the set in `excluded`, and returns
/// the result.
#[inline]
#[target_feature(enable = "mte")]
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub unsafe fn __arm_mte_exclude_tag<T>(src: *const T, excluded: u64) -> u64 {
    gmi_(src as *const (), excluded as i64) as u64
}

/// Store an allocation tag for the 16-byte granule of memory.
///
/// `tag_address`: A pointer containing an address and a logical tag, which
///                must be 16-byte aligned.
///
/// SAFETY: `tag_address` must be 16-byte aligned. The tag will apply to the
/// entire 16-byte memory granule.
#[inline]
#[target_feature(enable = "mte")]
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub unsafe fn __arm_mte_set_tag<T>(tag_address: *const T) {
    stg_(tag_address as *const (), tag_address as *const ());
}

/// Load an allocation tag from memory, returning a new pointer with the
/// corresponding logical tag.
///
/// `address`: A pointer containing an address from which allocation tag memory
///            is read. This does not need to be 16-byte aligned.
#[inline]
#[target_feature(enable = "mte")]
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub unsafe fn __arm_mte_get_tag<T>(address: *const T) -> *const T {
    ldg_(address as *const (), address as *const ()) as *const T
}

/// Calculate the difference between the address parts of two pointers, ignoring
/// the tags, and sign-extending the result.
#[inline]
#[target_feature(enable = "mte")]
#[unstable(feature = "stdarch_aarch64_mte", issue = "129010")]
pub unsafe fn __arm_mte_ptrdiff<T, U>(a: *const T, b: *const U) -> i64 {
    subp_(a as *const (), b as *const ())
}

#[cfg(test)]
mod test {
    use super::*;
    use stdarch_test::assert_instr;

    #[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(irg))] // FIXME: MSVC  `dumpbin` doesn't support MTE
    #[allow(dead_code)]
    #[target_feature(enable = "mte")]
    unsafe fn test_arm_mte_create_random_tag(src: *const (), mask: u64) -> *const () {
        __arm_mte_create_random_tag(src, mask)
    }

    #[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(addg))]
    #[allow(dead_code)]
    #[target_feature(enable = "mte")]
    unsafe fn test_arm_mte_increment_tag(src: *const ()) -> *const () {
        __arm_mte_increment_tag::<1, _>(src)
    }

    #[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(gmi))]
    #[allow(dead_code)]
    #[target_feature(enable = "mte")]
    unsafe fn test_arm_mte_exclude_tag(src: *const (), excluded: u64) -> u64 {
        __arm_mte_exclude_tag(src, excluded)
    }

    #[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(stg))]
    #[allow(dead_code)]
    #[target_feature(enable = "mte")]
    unsafe fn test_arm_mte_set_tag(src: *const ()) {
        __arm_mte_set_tag(src)
    }

    #[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(ldg))]
    #[allow(dead_code)]
    #[target_feature(enable = "mte")]
    unsafe fn test_arm_mte_get_tag(src: *const ()) -> *const () {
        __arm_mte_get_tag(src)
    }

    #[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(subp))]
    #[allow(dead_code)]
    #[target_feature(enable = "mte")]
    unsafe fn test_arm_mte_ptrdiff(a: *const (), b: *const ()) -> i64 {
        __arm_mte_ptrdiff(a, b)
    }
}
