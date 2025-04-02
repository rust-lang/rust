use crate::{mem::SizedTypeProperties, cfg_match, intrinsics};

/// Performs a volatile read of the value from `src` without moving it. This
/// leaves the memory in `src` unchanged.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `read_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must be properly aligned.
///
/// * `src` must point to a properly initialized value of type `T`.
///
/// Like [`read`], `read_volatile` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
/// However, storing non-[`Copy`] types in volatile memory is almost certainly
/// incorrect.
///
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
///
/// [valid]: self#safety
/// [read-ownership]: read#ownership-of-the-returned-value
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `read_volatile` and any write operation to the same location
/// is undefined behavior.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
#[cfg_attr(miri, track_caller)] // Even without panics, this helps for Miri backtraces
#[rustc_diagnostic_item = "ptr_read_volatile"]
pub unsafe fn read_volatile<T>(src: *const T) -> T {
    // SAFETY: the caller must uphold the safety contract for `volatile_load`.
    unsafe {
        crate::ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::read_volatile requires that the pointer argument is aligned and non-null",
            (
                addr: *const () = src as *const (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => crate::ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
        );
        cfg_match! {
            all(target_arch = "arm", target_feature = "thumb-mode", target_pointer_width = "32") => {
                {
                    use crate::arch::asm;
                    use crate::mem::MaybeUninit;

                    match size_of::<T>() {
                        // For the relevant sizes, ensure that just a single load is emitted
                        // for the read with nothing merged or split.
                        1 => {
                            let byte: MaybeUninit::<u8>;
                            asm!(
                                "ldrb {out}, [{in}]",
                                in = in(reg) src,
                                out = out(reg) byte
                            );

                            intrinsics::transmute_unchecked(byte)
                        }
                        2 => {
                            let halfword: MaybeUninit::<u16>;
                            asm!(
                                "ldrh {out}, [{in}]",
                                in = in(reg) src,
                                out = out(reg) halfword
                            );

                            intrinsics::transmute_unchecked(halfword)
                        },
                        4 => {
                            let word: MaybeUninit::<u32>;
                            asm!(
                                "ldr {out}, [{in}]",
                                in = in(reg) src,
                                out = out(reg) word
                            );

                            intrinsics::transmute_unchecked(word)
                        },
                        // Anything else is mostly meaningless.
                        _ => intrinsics::volatile_load(src),
                    }
            }}
            _ => {
                intrinsics::volatile_load(src)
            }
        }
    }
}
