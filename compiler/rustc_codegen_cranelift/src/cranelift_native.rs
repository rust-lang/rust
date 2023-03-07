// Vendored from https://github.com/bytecodealliance/wasmtime/blob/b58a197d33f044193c3d608010f5e6ec394ac07e/cranelift/native/src/lib.rs
// which is licensed as
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// unlike rustc_codegen_cranelift itself. Also applies a small change to remove #![cfg_attr] that
// rust's CI complains about and to fix formatting to match rustc.
// FIXME revert back to the external crate with Cranelift 0.93
#![allow(warnings)]

//! Performs autodetection of the host for the purposes of running
//! Cranelift to generate code to run on the same machine.

#![deny(missing_docs, trivial_numeric_casts, unused_extern_crates, unstable_features)]
#![warn(unused_import_braces)]

use cranelift_codegen::isa;
use target_lexicon::Triple;

/// Return an `isa` builder configured for the current host
/// machine, or `Err(())` if the host machine is not supported
/// in the current configuration.
pub fn builder() -> Result<isa::Builder, &'static str> {
    builder_with_options(true)
}

/// Return an `isa` builder configured for the current host
/// machine, or `Err(())` if the host machine is not supported
/// in the current configuration.
///
/// Selects the given backend variant specifically; this is
/// useful when more than oen backend exists for a given target
/// (e.g., on x86-64).
pub fn builder_with_options(infer_native_flags: bool) -> Result<isa::Builder, &'static str> {
    let mut isa_builder = isa::lookup(Triple::host()).map_err(|err| match err {
        isa::LookupError::SupportDisabled => "support for architecture disabled at compile time",
        isa::LookupError::Unsupported => "unsupported architecture",
    })?;

    #[cfg(target_arch = "x86_64")]
    {
        use cranelift_codegen::settings::Configurable;

        if !std::is_x86_feature_detected!("sse2") {
            return Err("x86 support requires SSE2");
        }

        if !infer_native_flags {
            return Ok(isa_builder);
        }

        // These are temporarily enabled by default (see #3810 for
        // more) so that a default-constructed `Flags` can work with
        // default Wasmtime features. Otherwise, the user must
        // explicitly use native flags or turn these on when on x86-64
        // platforms to avoid a configuration panic. In order for the
        // "enable if detected" logic below to work, we must turn them
        // *off* (differing from the default) and then re-enable below
        // if present.
        isa_builder.set("has_sse3", "false").unwrap();
        isa_builder.set("has_ssse3", "false").unwrap();
        isa_builder.set("has_sse41", "false").unwrap();
        isa_builder.set("has_sse42", "false").unwrap();

        if std::is_x86_feature_detected!("sse3") {
            isa_builder.enable("has_sse3").unwrap();
        }
        if std::is_x86_feature_detected!("ssse3") {
            isa_builder.enable("has_ssse3").unwrap();
        }
        if std::is_x86_feature_detected!("sse4.1") {
            isa_builder.enable("has_sse41").unwrap();
        }
        if std::is_x86_feature_detected!("sse4.2") {
            isa_builder.enable("has_sse42").unwrap();
        }
        if std::is_x86_feature_detected!("popcnt") {
            isa_builder.enable("has_popcnt").unwrap();
        }
        if std::is_x86_feature_detected!("avx") {
            isa_builder.enable("has_avx").unwrap();
        }
        if std::is_x86_feature_detected!("avx2") {
            isa_builder.enable("has_avx2").unwrap();
        }
        if std::is_x86_feature_detected!("fma") {
            isa_builder.enable("has_fma").unwrap();
        }
        if std::is_x86_feature_detected!("bmi1") {
            isa_builder.enable("has_bmi1").unwrap();
        }
        if std::is_x86_feature_detected!("bmi2") {
            isa_builder.enable("has_bmi2").unwrap();
        }
        if std::is_x86_feature_detected!("avx512bitalg") {
            isa_builder.enable("has_avx512bitalg").unwrap();
        }
        if std::is_x86_feature_detected!("avx512dq") {
            isa_builder.enable("has_avx512dq").unwrap();
        }
        if std::is_x86_feature_detected!("avx512f") {
            isa_builder.enable("has_avx512f").unwrap();
        }
        if std::is_x86_feature_detected!("avx512vl") {
            isa_builder.enable("has_avx512vl").unwrap();
        }
        if std::is_x86_feature_detected!("avx512vbmi") {
            isa_builder.enable("has_avx512vbmi").unwrap();
        }
        if std::is_x86_feature_detected!("lzcnt") {
            isa_builder.enable("has_lzcnt").unwrap();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use cranelift_codegen::settings::Configurable;

        if !infer_native_flags {
            return Ok(isa_builder);
        }

        if std::arch::is_aarch64_feature_detected!("lse") {
            isa_builder.enable("has_lse").unwrap();
        }

        if std::arch::is_aarch64_feature_detected!("paca") {
            isa_builder.enable("has_pauth").unwrap();
        }

        if cfg!(target_os = "macos") {
            // Pointer authentication is always available on Apple Silicon.
            isa_builder.enable("sign_return_address").unwrap();
            // macOS enforces the use of the B key for return addresses.
            isa_builder.enable("sign_return_address_with_bkey").unwrap();
        }
    }

    // There is no is_s390x_feature_detected macro yet, so for now
    // we use getauxval from the libc crate directly.
    #[cfg(all(target_arch = "s390x", target_os = "linux"))]
    {
        use cranelift_codegen::settings::Configurable;

        if !infer_native_flags {
            return Ok(isa_builder);
        }

        let v = unsafe { libc::getauxval(libc::AT_HWCAP) };
        const HWCAP_S390X_VXRS_EXT2: libc::c_ulong = 32768;
        if (v & HWCAP_S390X_VXRS_EXT2) != 0 {
            isa_builder.enable("has_vxrs_ext2").unwrap();
            // There is no separate HWCAP bit for mie2, so assume
            // that any machine with vxrs_ext2 also has mie2.
            isa_builder.enable("has_mie2").unwrap();
        }
    }

    // `is_riscv_feature_detected` is nightly only for now, use
    // getauxval from the libc crate directly as a temporary measure.
    #[cfg(all(target_arch = "riscv64", target_os = "linux"))]
    {
        use cranelift_codegen::settings::Configurable;

        if !infer_native_flags {
            return Ok(isa_builder);
        }

        let v = unsafe { libc::getauxval(libc::AT_HWCAP) };

        const HWCAP_RISCV_EXT_A: libc::c_ulong = 1 << (b'a' - b'a');
        const HWCAP_RISCV_EXT_C: libc::c_ulong = 1 << (b'c' - b'a');
        const HWCAP_RISCV_EXT_D: libc::c_ulong = 1 << (b'd' - b'a');
        const HWCAP_RISCV_EXT_F: libc::c_ulong = 1 << (b'f' - b'a');
        const HWCAP_RISCV_EXT_M: libc::c_ulong = 1 << (b'm' - b'a');
        const HWCAP_RISCV_EXT_V: libc::c_ulong = 1 << (b'v' - b'a');

        if (v & HWCAP_RISCV_EXT_A) != 0 {
            isa_builder.enable("has_a").unwrap();
        }

        if (v & HWCAP_RISCV_EXT_C) != 0 {
            isa_builder.enable("has_c").unwrap();
        }

        if (v & HWCAP_RISCV_EXT_D) != 0 {
            isa_builder.enable("has_d").unwrap();
        }

        if (v & HWCAP_RISCV_EXT_F) != 0 {
            isa_builder.enable("has_f").unwrap();

            // TODO: There doesn't seem to be a bit associated with this extension
            // rust enables it with the `f` extension:
            // https://github.com/rust-lang/stdarch/blob/790411f93c4b5eada3c23abb4c9a063fb0b24d99/crates/std_detect/src/detect/os/linux/riscv.rs#L43
            isa_builder.enable("has_zicsr").unwrap();
        }

        if (v & HWCAP_RISCV_EXT_M) != 0 {
            isa_builder.enable("has_m").unwrap();
        }

        if (v & HWCAP_RISCV_EXT_V) != 0 {
            isa_builder.enable("has_v").unwrap();
        }

        // TODO: ZiFencei does not have a bit associated with it
        // TODO: Zbkb does not have a bit associated with it
    }

    // squelch warnings about unused mut/variables on some platforms.
    drop(&mut isa_builder);
    drop(infer_native_flags);

    Ok(isa_builder)
}

#[cfg(test)]
mod tests {
    use super::builder;
    use cranelift_codegen::isa::CallConv;
    use cranelift_codegen::settings;

    #[test]
    fn test() {
        if let Ok(isa_builder) = builder() {
            let flag_builder = settings::builder();
            let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                assert_eq!(isa.default_call_conv(), CallConv::AppleAarch64);
            } else if cfg!(any(unix, target_os = "nebulet")) {
                assert_eq!(isa.default_call_conv(), CallConv::SystemV);
            } else if cfg!(windows) {
                assert_eq!(isa.default_call_conv(), CallConv::WindowsFastcall);
            }

            if cfg!(target_pointer_width = "64") {
                assert_eq!(isa.pointer_bits(), 64);
            } else if cfg!(target_pointer_width = "32") {
                assert_eq!(isa.pointer_bits(), 32);
            } else if cfg!(target_pointer_width = "16") {
                assert_eq!(isa.pointer_bits(), 16);
            }
        }
    }
}

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
