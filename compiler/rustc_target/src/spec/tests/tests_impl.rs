use super::super::*;
use std::assert_matches::assert_matches;

// Test target self-consistency and JSON encoding/decoding roundtrip.
pub(super) fn test_target(target: Target) {
    target.check_consistency();
    assert_eq!(Target::from_json(target.to_json()).map(|(j, _)| j), Ok(target));
}

impl Target {
    fn check_consistency(&self) {
        assert_eq!(self.is_like_osx, self.vendor == "apple");
        assert_eq!(self.is_like_solaris, self.os == "solaris" || self.os == "illumos");
        assert_eq!(self.is_like_windows, self.os == "windows" || self.os == "uefi");
        assert_eq!(self.is_like_wasm, self.arch == "wasm32" || self.arch == "wasm64");
        assert!(self.is_like_windows || !self.is_like_msvc);

        // Check that default linker flavor and lld flavor are compatible
        // with some other key properties.
        assert_eq!(self.is_like_osx, matches!(self.lld_flavor, LldFlavor::Ld64));
        assert_eq!(self.is_like_msvc, matches!(self.lld_flavor, LldFlavor::Link));
        assert_eq!(self.is_like_wasm, matches!(self.lld_flavor, LldFlavor::Wasm));
        assert_eq!(self.os == "l4re", matches!(self.linker_flavor, LinkerFlavor::L4Bender));
        assert_eq!(self.os == "emscripten", matches!(self.linker_flavor, LinkerFlavor::Em));
        assert_eq!(self.arch == "bpf", matches!(self.linker_flavor, LinkerFlavor::BpfLinker));
        assert_eq!(self.arch == "nvptx64", matches!(self.linker_flavor, LinkerFlavor::PtxLinker));

        for args in [
            &self.pre_link_args,
            &self.late_link_args,
            &self.late_link_args_dynamic,
            &self.late_link_args_static,
            &self.post_link_args,
        ] {
            for (&flavor, flavor_args) in args {
                assert!(!flavor_args.is_empty());
                // Check that flavors mentioned in link args are compatible with the default flavor.
                match (self.linker_flavor, self.lld_flavor) {
                    (
                        LinkerFlavor::Ld | LinkerFlavor::Lld(LldFlavor::Ld) | LinkerFlavor::Gcc,
                        LldFlavor::Ld,
                    ) => {
                        assert_matches!(
                            flavor,
                            LinkerFlavor::Ld | LinkerFlavor::Lld(LldFlavor::Ld) | LinkerFlavor::Gcc
                        )
                    }
                    (LinkerFlavor::Gcc, LldFlavor::Ld64) => {
                        assert_matches!(
                            flavor,
                            LinkerFlavor::Lld(LldFlavor::Ld64) | LinkerFlavor::Gcc
                        )
                    }
                    (LinkerFlavor::Msvc | LinkerFlavor::Lld(LldFlavor::Link), LldFlavor::Link) => {
                        assert_matches!(
                            flavor,
                            LinkerFlavor::Msvc | LinkerFlavor::Lld(LldFlavor::Link)
                        )
                    }
                    (LinkerFlavor::Lld(LldFlavor::Wasm) | LinkerFlavor::Gcc, LldFlavor::Wasm) => {
                        assert_matches!(
                            flavor,
                            LinkerFlavor::Lld(LldFlavor::Wasm) | LinkerFlavor::Gcc
                        )
                    }
                    (LinkerFlavor::L4Bender, LldFlavor::Ld) => {
                        assert_matches!(flavor, LinkerFlavor::L4Bender)
                    }
                    (LinkerFlavor::Em, LldFlavor::Wasm) => {
                        assert_matches!(flavor, LinkerFlavor::Em)
                    }
                    (LinkerFlavor::BpfLinker, LldFlavor::Ld) => {
                        assert_matches!(flavor, LinkerFlavor::BpfLinker)
                    }
                    (LinkerFlavor::PtxLinker, LldFlavor::Ld) => {
                        assert_matches!(flavor, LinkerFlavor::PtxLinker)
                    }
                    flavors => unreachable!("unexpected flavor combination: {:?}", flavors),
                }

                // Check that link args for cc and non-cc versions of flavors are consistent.
                let check_noncc = |noncc_flavor| {
                    if let Some(noncc_args) = args.get(&noncc_flavor) {
                        for arg in flavor_args {
                            if let Some(suffix) = arg.strip_prefix("-Wl,") {
                                assert!(noncc_args.iter().any(|a| a == suffix));
                            }
                        }
                    }
                };
                match self.linker_flavor {
                    LinkerFlavor::Gcc => match self.lld_flavor {
                        LldFlavor::Ld => {
                            check_noncc(LinkerFlavor::Ld);
                            check_noncc(LinkerFlavor::Lld(LldFlavor::Ld));
                        }
                        LldFlavor::Wasm => check_noncc(LinkerFlavor::Lld(LldFlavor::Wasm)),
                        LldFlavor::Ld64 | LldFlavor::Link => {}
                    },
                    _ => {}
                }
            }

            // Check that link args for lld and non-lld versions of flavors are consistent.
            assert_eq!(args.get(&LinkerFlavor::Ld), args.get(&LinkerFlavor::Lld(LldFlavor::Ld)));
            assert_eq!(
                args.get(&LinkerFlavor::Msvc),
                args.get(&LinkerFlavor::Lld(LldFlavor::Link)),
            );
        }

        assert!(
            (self.pre_link_objects_self_contained.is_empty()
                && self.post_link_objects_self_contained.is_empty())
                || self.link_self_contained != LinkSelfContainedDefault::False
        );

        // If your target really needs to deviate from the rules below,
        // except it and document the reasons.
        // Keep the default "unknown" vendor instead.
        assert_ne!(self.vendor, "");
        if !self.can_use_os_unknown() {
            // Keep the default "none" for bare metal targets instead.
            assert_ne!(self.os, "unknown");
        }
    }

    // Add your target to the whitelist if it has `std` library
    // and you certainly want "unknown" for the OS name.
    fn can_use_os_unknown(&self) -> bool {
        self.llvm_target == "wasm32-unknown-unknown"
            || self.llvm_target == "wasm64-unknown-unknown"
            || (self.env == "sgx" && self.vendor == "fortanix")
    }
}
