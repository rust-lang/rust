// Armv6k supports atomic instructions, but they are unavailable in Thumb mode
// unless Thumb-2 instructions available (v6t2).
// Using Thumb interworking allows us to use these instructions even from Thumb mode
// without Thumb-2 instructions, but LLVM does not implement that processing (as of LLVM 21),
// so we implement it here at this time.

use core::arch::asm;
use core::mem;

// Data Memory Barrier (DMB) operation.
//
// Armv6 does not support DMB instruction, so use use special instruction equivalent to it.
//
// Refs: https://developer.arm.com/documentation/ddi0360/f/control-coprocessor-cp15/register-descriptions/c7--cache-operations-register
macro_rules! cp15_barrier {
    () => {
        "mcr p15, #0, {zero}, c7, c10, #5"
    };
}

#[instruction_set(arm::a32)]
unsafe fn fence() {
    unsafe {
        asm!(
            cp15_barrier!(),
            // cp15_barrier! calls `mcr p15, 0, {zero}, c7, c10, 5`, and
            // the value in the {zero} register should be zero (SBZ).
            zero = inout(reg) 0_u32 => _,
            options(nostack, preserves_flags),
        );
    }
}

trait Atomic: Copy + Eq {
    unsafe fn load_relaxed(src: *const Self) -> Self;
    unsafe fn cmpxchg(dst: *mut Self, current: Self, new: Self) -> Self;
}

macro_rules! atomic {
    ($ty:ident, $suffix:tt) => {
        impl Atomic for $ty {
            // #[instruction_set(arm::a32)] is unneeded for ldr.
            #[inline]
            unsafe fn load_relaxed(
                src: *const Self,
            ) -> Self {
                let out: Self;
                // SAFETY: the caller must guarantee that the pointer is valid for read and write
                // and aligned to the element size.
                unsafe {
                    asm!(
                        concat!("ldr", $suffix, " {out}, [{src}]"), // atomic { out = *src }
                        src = in(reg) src,
                        out = lateout(reg) out,
                        options(nostack, preserves_flags),
                    );
                }
                out
            }
            #[inline]
            #[instruction_set(arm::a32)]
            unsafe fn cmpxchg(
                dst: *mut Self,
                old: Self,
                new: Self,
            ) -> Self {
                let mut out: Self;
                // SAFETY: the caller must guarantee that the pointer is valid for read and write
                // and aligned to the element size.
                //
                // Instead of the common `fence; ll/sc loop; fence` form, we use the form used by
                // LLVM, which omits the preceding fence if no write operation is performed.
                unsafe {
                    asm!(
                            concat!("ldrex", $suffix, " {out}, [{dst}]"),      // atomic { out = *dst; EXCLUSIVE = dst }
                            "cmp {out}, {old}",                                // if out == old { Z = 1 } else { Z = 0 }
                            "bne 3f",                                          // if Z == 0 { jump 'cmp-fail }
                            cp15_barrier!(),                                            // fence
                        "2:", // 'retry:
                            concat!("strex", $suffix, " {r}, {new}, [{dst}]"), // atomic { if EXCLUSIVE == dst { *dst = new; r = 0 } else { r = 1 }; EXCLUSIVE = None }
                            "cmp {r}, #0",                                     // if r == 0 { Z = 1 } else { Z = 0 }
                            "beq 3f",                                          // if Z == 1 { jump 'success }
                            concat!("ldrex", $suffix, " {out}, [{dst}]"),      // atomic { out = *dst; EXCLUSIVE = dst }
                            "cmp {out}, {old}",                                // if out == old { Z = 1 } else { Z = 0 }
                            "beq 2b",                                          // if Z == 1 { jump 'retry }
                        "3:", // 'cmp-fail | 'success:
                            cp15_barrier!(),                                            // fence
                        dst = in(reg) dst,
                        // Note: this cast must be a zero-extend since loaded value
                        // which compared to it is zero-extended.
                        old = in(reg) u32::from(old),
                        new = in(reg) new,
                        out = out(reg) out,
                        r = out(reg) _,
                        // cp15_barrier! calls `mcr p15, 0, {zero}, c7, c10, 5`, and
                        // the value in the {zero} register should be zero (SBZ).
                        zero = inout(reg) 0_u32 => _,
                        // Do not use `preserves_flags` because CMP modifies the condition flags.
                        options(nostack),
                    );
                    out
                }
            }
        }
    };
}
atomic!(u8, "b");
atomic!(u16, "h");
atomic!(u32, "");

// To avoid the annoyance of sign extension, we implement signed CAS using
// unsigned CAS. (See note in cmpxchg impl in atomic! macro)
macro_rules! delegate_signed {
    ($ty:ident, $base:ident) => {
        const _: () = {
            assert!(mem::size_of::<$ty>() == mem::size_of::<$base>());
            assert!(mem::align_of::<$ty>() == mem::align_of::<$base>());
        };
        impl Atomic for $ty {
            #[inline]
            unsafe fn load_relaxed(src: *const Self) -> Self {
                // SAFETY: the caller must uphold the safety contract.
                // casts are okay because $ty and $base implement the same layout.
                unsafe { <$base as Atomic>::load_relaxed(src.cast::<$base>()).cast_signed() }
            }
            #[inline]
            unsafe fn cmpxchg(dst: *mut Self, old: Self, new: Self) -> Self {
                // SAFETY: the caller must uphold the safety contract.
                // casts are okay because $ty and $base implement the same layout.
                unsafe {
                    <$base as Atomic>::cmpxchg(
                        dst.cast::<$base>(),
                        old.cast_unsigned(),
                        new.cast_unsigned(),
                    )
                    .cast_signed()
                }
            }
        }
    };
}
delegate_signed!(i8, u8);
delegate_signed!(i16, u16);
delegate_signed!(i32, u32);

// Generic atomic read-modify-write operation
//
// We could implement RMW more efficiently as an assembly LL/SC loop per operation,
// but we won't do that for now because it would make the implementation more complex.
//
// We also do not implement LL and SC as separate functions. This is because it
// is theoretically possible for the compiler to insert operations that might
// clear the reservation between LL and SC. See https://github.com/taiki-e/portable-atomic/blob/58ef7f27c9e20da4cc1ef0abf8b8ce9ac5219ec3/src/imp/atomic128/aarch64.rs#L44-L55
// for more details.
unsafe fn atomic_rmw<T: Atomic, F: Fn(T) -> T, G: Fn(T, T) -> T>(ptr: *mut T, f: F, g: G) -> T {
    loop {
        // SAFETY: the caller must guarantee that the pointer is valid for read and write
        // and aligned to the element size.
        let curval = unsafe { T::load_relaxed(ptr) };
        let newval = f(curval);
        // SAFETY: the caller must guarantee that the pointer is valid for read and write
        // and aligned to the element size.
        if unsafe { T::cmpxchg(ptr, curval, newval) } == curval {
            return g(curval, newval);
        }
    }
}

macro_rules! atomic_rmw {
    ($name:ident, $ty:ty, $op:expr, $fetch:expr) => {
        intrinsics! {
            pub unsafe extern "C" fn $name(ptr: *mut $ty, val: $ty) -> $ty {
                // SAFETY: the caller must guarantee that the pointer is valid for read and write
                // and aligned to the element size.
                unsafe {
                    atomic_rmw(
                        ptr,
                        |x| $op(x as $ty, val),
                        |old, new| $fetch(old, new)
                    ) as $ty
                }
            }
        }
    };

    (@old $name:ident, $ty:ty, $op:expr) => {
        atomic_rmw!($name, $ty, $op, |old, _| old);
    };

    (@new $name:ident, $ty:ty, $op:expr) => {
        atomic_rmw!($name, $ty, $op, |_, new| new);
    };
}
macro_rules! atomic_cmpxchg {
    ($name:ident, $ty:ty) => {
        intrinsics! {
            pub unsafe extern "C" fn $name(ptr: *mut $ty, oldval: $ty, newval: $ty) -> $ty {
                // SAFETY: the caller must guarantee that the pointer is valid for read and write
                // and aligned to the element size.
                unsafe { <$ty as Atomic>::cmpxchg(ptr, oldval, newval) }
            }
        }
    };
}

include!("arm_thumb_shared.rs");

intrinsics! {
    pub unsafe extern "C" fn __sync_synchronize() {
       // SAFETY: preconditions are the same as the calling function.
       unsafe { fence() };
    }
}
