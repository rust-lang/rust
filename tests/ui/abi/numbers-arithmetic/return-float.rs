//@ run-pass
//@ revisions: no-opts cg-opts-only all-opts
// No optimisations: Functions won't be inlined, so the machine-level ABI will be tested.
//@[no-opts] compile-flags: -Copt-level=0
// Codegen optimisations only: Functions will only be inlined by the codegen backend.
//@[cg-opts-only] compile-flags: -Copt-level=3 -Zmir-opt-level=0
// All optimisations: Functions will be inlined in MIR or the codegen backend.
//@[all-opts] compile-flags: -Copt-level=3

#![cfg_attr(all(target_arch = "x86", target_feature = "sse2"), feature(abi_vectorcall))]

// Test that floats (in particular signalling NaNs) are losslessly returned from functions.

use std::mem::MaybeUninit;

#[link(name = "rust_test_helpers", kind = "static")]
unsafe extern "C" {
    safe fn rust_dbg_extern_identity_float(x: f32) -> f32;
    safe fn rust_dbg_extern_identity_double(x: f64) -> f64;
    safe fn rust_dbg_extern_call_float(
        f: extern "C" fn(f32) -> f32,
        x: f32,
        res: &mut MaybeUninit<f32>,
    );
    safe fn rust_dbg_extern_call_double(
        f: extern "C" fn(f64) -> f64,
        x: f64,
        res: &mut MaybeUninit<f64>,
    );
}

fn main() {
    // FIXME(f16,f128): Add test cases for `f16` and `f128`.
    let bits_f32 = std::hint::black_box(
        const {
            [
                4.2_f32.to_bits(),
                f32::INFINITY.to_bits(),
                f32::NEG_INFINITY.to_bits(),
                f32::NAN.to_bits(),
                // These two masks cover all the fraction bits. One of them is a signalling NaN, the
                // other is quiet.
                // Similar to the masks in `test_float_bits_conv` in library/std/src/f32/tests.rs
                f32::NAN.to_bits() ^ 0x002A_AAAA,
                f32::NAN.to_bits() ^ 0x0055_5555,
                // Same as above but with the sign bit flipped.
                f32::NAN.to_bits() ^ 0x802A_AAAA,
                f32::NAN.to_bits() ^ 0x8055_5555,
            ]
        },
    );
    let bits_f64 = std::hint::black_box(
        const {
            [
                4.2_f64.to_bits(),
                f64::INFINITY.to_bits(),
                f64::NEG_INFINITY.to_bits(),
                f64::NAN.to_bits(),
                // These two masks cover all the fraction bits. One of them is a signalling NaN, the
                // other is quiet.
                // Similar to the masks in `test_float_bits_conv` in library/std/src/f64/tests.rs
                f64::NAN.to_bits() ^ 0x000A_AAAA_AAAA_AAAA,
                f64::NAN.to_bits() ^ 0x0005_5555_5555_5555,
                // Same as above but with the sign bit flipped.
                f64::NAN.to_bits() ^ 0x800A_AAAA_AAAA_AAAA,
                f64::NAN.to_bits() ^ 0x8005_5555_5555_5555,
            ]
        },
    );

    #[repr(C)]
    struct Struct<T>(T);

    let x86 = cfg!(target_arch = "x86");
    let x86_no_sse = cfg!(all(target_arch = "x86", not(target_feature = "sse")));
    let x86_no_sse2 = cfg!(all(target_arch = "x86", not(target_feature = "sse2")));
    // On 32-bit x86, `f32`s are returned on the x87 stack. `external` allows the NaN to be
    // quietened in that case, as the C compiler might not have ensured that placing the float on
    // the the stack is lossless.
    // FIXME(#114479): LLVM miscompiles loading and storing `f32` and `f64` when SSE(2) is disabled
    // on x86 (i586-* targets), meaning signalling NaNs get quietened in the middle of function
    // bodies.
    let check_f32 = |res: u32, bits: u32, abi: &str, what: &str, external: bool| {
        let quiet_bit = 0x0040_0000;
        assert!(
            res == bits
                || (x86
                    && (external || x86_no_sse)
                    && f32::from_bits(bits).is_nan()
                    && bits | quiet_bit == res),
            "{res:x} != {bits:x} {} f32{} {}",
            abi,
            if external { " extern" } else { "" },
            what
        );
    };
    let check_f64 = |res: u64, bits: u64, abi: &str, what: &str, external: bool| {
        let quiet_bit = 0x0008_0000_0000_0000;
        assert!(
            res == bits
                || (x86
                    && (external || x86_no_sse2)
                    && f64::from_bits(bits).is_nan()
                    && bits | quiet_bit == res),
            "{res:x} != {bits:x} {} f64{} {}",
            abi,
            if external { " extern" } else { "" },
            what
        );
    };

    macro_rules! abi_check {
        ($($abi:literal),+ $(,)?) => {
            $({
                #[cfg_attr(no_opts, inline(never))]
                extern $abi fn identity<T>(x: T) -> T {
                    x
                }

                for bits in bits_f32 {
                    check_f32(
                        identity(f32::from_bits(bits)).to_bits(),
                        bits,
                        $abi,
                        "direct",
                        false,
                    );
                    // Check single element structs are returned correctly too.
                    check_f32(
                        identity(Struct(f32::from_bits(bits))).0.to_bits(),
                        bits,
                        $abi,
                        "struct",
                        false,
                    );
                    // Ensure value is still preserved when wrapped in a MaybeUninit.
                    unsafe {
                        check_f32(
                            identity(MaybeUninit::new(f32::from_bits(bits)))
                                .assume_init()
                                .to_bits(),
                            bits,
                            $abi,
                            "MaybeUninit",
                            false,
                        );
                        check_f32(
                            identity(MaybeUninit::new(Struct(f32::from_bits(bits))))
                                .assume_init()
                                .0
                                .to_bits(),
                            bits,
                            $abi,
                            "struct MaybeUninit",
                            false,
                        );
                    }
                }
                for bits in bits_f64 {
                    check_f64(
                        identity(f64::from_bits(bits)).to_bits(),
                        bits,
                        $abi,
                        "direct",
                        false,
                    );
                    check_f64(
                        identity(Struct(f64::from_bits(bits))).0.to_bits(),
                        bits,
                        $abi,
                        "struct",
                        false,
                    );
                    unsafe {
                        check_f64(
                            identity(MaybeUninit::new(f64::from_bits(bits)))
                                .assume_init()
                                .to_bits(),
                            bits,
                            $abi,
                            "MaybeUninit",
                            false,
                        );
                        check_f64(
                            identity(MaybeUninit::new(Struct(f64::from_bits(bits))))
                                .assume_init()
                                .0
                                .to_bits(),
                            bits,
                            $abi,
                            "struct MaybeUninit",
                            false,
                        );
                    }
                }
                // Returning `MaybeUninit::uninit()` must not cause undefined behaviour.
                std::hint::black_box(identity(MaybeUninit::<f32>::uninit()));
                std::hint::black_box(identity(MaybeUninit::<Struct<f32>>::uninit()));
                std::hint::black_box(identity(MaybeUninit::<f32>::uninit()));
                std::hint::black_box(identity(MaybeUninit::<Struct<f64>>::uninit()));
            })*
        };
    }
    abi_check!("Rust", "C", "C-unwind", "system", "system-unwind");
    // Test some extra platform-specific ABIs on 32-bit x86 as that is the platform where signaling
    // NaNs often used to get quietened (and still do get quietened when SSE2 is disabled).
    #[cfg(target_arch = "x86")]
    abi_check!(
        "cdecl",
        "cdecl-unwind",
        "efiapi",
        "fastcall",
        "fastcall-unwind",
        "stdcall",
        "stdcall-unwind",
        "thiscall",
        "thiscall-unwind",
    );
    #[cfg(all(target_arch = "x86", target_feature = "sse2"))]
    abi_check!("vectorcall", "vectorcall-unwind");

    // Test types that are returned as scalar pairs.
    #[cfg_attr(no_opts, inline(never))]
    fn identity<T>(x: T) -> T {
        x
    }

    for bits in bits_f32 {
        check_f32(identity((f32::from_bits(bits), 42)).0.to_bits(), bits, "Rust", "tuple.0", false);
        check_f32(identity((42, f32::from_bits(bits))).1.to_bits(), bits, "Rust", "tuple.1", false);
        let (a, b) = identity((f32::from_bits(bits), f32::from_bits(bits)));
        check_f32(a.to_bits(), bits, "Rust", "pair.0", false);
        check_f32(b.to_bits(), bits, "Rust", "pair.1", false);
    }
    for bits in bits_f64 {
        check_f64(identity((f64::from_bits(bits), 42)).0.to_bits(), bits, "Rust", "tuple.0", false);
        check_f64(identity((42, f64::from_bits(bits))).1.to_bits(), bits, "Rust", "tuple.1", false);
        let (a, b) = identity((f64::from_bits(bits), f64::from_bits(bits)));
        check_f64(a.to_bits(), bits, "Rust", "pair.0", false);
        check_f64(b.to_bits(), bits, "Rust", "pair.1", false);
    }

    // Test calling and being called by a C function.
    #[cfg_attr(no_opts, inline(never))]
    extern "C" fn identity_c<T>(x: T) -> T {
        x
    }

    for bits in bits_f32 {
        let res = rust_dbg_extern_identity_float(f32::from_bits(bits)).to_bits();
        // On 32-bit x86, `f32`s are returned on the x87 stack. Allow the result to have been
        // quietened, as the C compiler might not have ensured that placing the float on the
        // the stack is lossless.
        check_f32(res, bits, "C", "Rust calling C", true);
        let res = unsafe {
            let mut res = MaybeUninit::uninit();
            rust_dbg_extern_call_float(identity_c, f32::from_bits(bits), &mut res);
            res.assume_init().to_bits()
        };
        check_f32(res, bits, "C", "C calling Rust", true);
    }
    for bits in bits_f64 {
        let res = rust_dbg_extern_identity_double(f64::from_bits(bits)).to_bits();
        // On 32-bit x86, `f64`s are returned on the x87 stack. Allow the result to have been
        // quietened, as the C compiler might not have ensured that placing the float on the
        // the stack is lossless.
        check_f64(res, bits, "C", "Rust calling C", true);
        let res = unsafe {
            let mut res = MaybeUninit::uninit();
            rust_dbg_extern_call_double(identity_c, f64::from_bits(bits), &mut res);
            res.assume_init().to_bits()
        };
        check_f64(res, bits, "C", "C calling Rust", true);
    }
}
