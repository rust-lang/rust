use super::super::*;
use std::assert_matches::assert_matches;

// Test target self-consistency and JSON encoding/decoding roundtrip.
pub(super) fn test_target(mut target: Target) {
    let recycled_target = Target::from_json(target.to_json()).map(|(j, _)| j);
    target.update_to_cli();
    target.check_consistency();
    assert_eq!(recycled_target, Ok(target));
}

impl Target {
    fn check_consistency(&self) {
        assert_eq!(self.is_like_osx, self.vendor == "apple");
        assert_eq!(self.is_like_solaris, self.os == "solaris" || self.os == "illumos");
        assert_eq!(self.is_like_windows, self.os == "windows" || self.os == "uefi");
        assert_eq!(self.is_like_wasm, self.arch == "wasm32" || self.arch == "wasm64");
        if self.is_like_msvc {
            assert!(self.is_like_windows);
        }

        // Check that default linker flavor is compatible with some other key properties.
        assert_eq!(self.is_like_osx, matches!(self.linker_flavor, LinkerFlavor::Darwin(..)));
        assert_eq!(self.is_like_msvc, matches!(self.linker_flavor, LinkerFlavor::Msvc(..)));
        assert_eq!(
            self.is_like_wasm && self.os != "emscripten",
            matches!(self.linker_flavor, LinkerFlavor::WasmLld(..))
        );
        assert_eq!(self.os == "emscripten", matches!(self.linker_flavor, LinkerFlavor::EmCc));
        assert_eq!(self.arch == "bpf", matches!(self.linker_flavor, LinkerFlavor::Bpf));
        assert_eq!(self.arch == "nvptx64", matches!(self.linker_flavor, LinkerFlavor::Ptx));

        for args in [
            &self.pre_link_args,
            &self.late_link_args,
            &self.late_link_args_dynamic,
            &self.late_link_args_static,
            &self.post_link_args,
        ] {
            for (&flavor, flavor_args) in &**args {
                assert!(!flavor_args.is_empty());
                // Check that flavors mentioned in link args are compatible with the default flavor.
                match self.linker_flavor {
                    LinkerFlavor::Gnu(..) => {
                        assert_matches!(flavor, LinkerFlavor::Gnu(..));
                    }
                    LinkerFlavor::Darwin(..) => {
                        assert_matches!(flavor, LinkerFlavor::Darwin(..))
                    }
                    LinkerFlavor::WasmLld(..) => {
                        assert_matches!(flavor, LinkerFlavor::WasmLld(..))
                    }
                    LinkerFlavor::Unix(..) => {
                        assert_matches!(flavor, LinkerFlavor::Unix(..));
                    }
                    LinkerFlavor::Msvc(..) => {
                        assert_matches!(flavor, LinkerFlavor::Msvc(..))
                    }
                    LinkerFlavor::EmCc
                    | LinkerFlavor::Bpf
                    | LinkerFlavor::Ptx
                    | LinkerFlavor::Llbc => {
                        assert_eq!(flavor, self.linker_flavor)
                    }
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
                    LinkerFlavor::Gnu(Cc::Yes, lld) => check_noncc(LinkerFlavor::Gnu(Cc::No, lld)),
                    LinkerFlavor::WasmLld(Cc::Yes) => check_noncc(LinkerFlavor::WasmLld(Cc::No)),
                    LinkerFlavor::Unix(Cc::Yes) => check_noncc(LinkerFlavor::Unix(Cc::No)),
                    _ => {}
                }
            }

            // Check that link args for lld and non-lld versions of flavors are consistent.
            for cc in [Cc::No, Cc::Yes] {
                assert_eq!(
                    args.get(&LinkerFlavor::Gnu(cc, Lld::No)),
                    args.get(&LinkerFlavor::Gnu(cc, Lld::Yes)),
                );
                assert_eq!(
                    args.get(&LinkerFlavor::Darwin(cc, Lld::No)),
                    args.get(&LinkerFlavor::Darwin(cc, Lld::Yes)),
                );
            }
            assert_eq!(
                args.get(&LinkerFlavor::Msvc(Lld::No)),
                args.get(&LinkerFlavor::Msvc(Lld::Yes)),
            );
        }

        if self.link_self_contained.is_disabled() {
            assert!(
                self.pre_link_objects_self_contained.is_empty()
                    && self.post_link_objects_self_contained.is_empty()
            );
        }

        // If your target really needs to deviate from the rules below,
        // except it and document the reasons.
        // Keep the default "unknown" vendor instead.
        assert_ne!(self.vendor, "");
        assert_ne!(self.os, "");
        if !self.can_use_os_unknown() {
            // Keep the default "none" for bare metal targets instead.
            assert_ne!(self.os, "unknown");
        }

        // Check dynamic linking stuff
        // BPF: when targeting user space vms (like rbpf), those can load dynamic libraries.
        // hexagon: when targeting QuRT, that OS can load dynamic libraries.
        if self.os == "none" && (self.arch != "bpf" && self.arch != "hexagon") {
            assert!(!self.dynamic_linking);
        }
        if self.only_cdylib
            || self.crt_static_allows_dylibs
            || !self.late_link_args_dynamic.is_empty()
        {
            assert!(self.dynamic_linking);
        }
        // Apparently PIC was slow on wasm at some point, see comments in wasm_base.rs
        if self.dynamic_linking && !(self.is_like_wasm && self.os != "emscripten") {
            assert_eq!(self.relocation_model, RelocModel::Pic);
        }
        if self.position_independent_executables {
            assert_eq!(self.relocation_model, RelocModel::Pic);
        }
        // The UEFI targets do not support dynamic linking but still require PIC (#101377).
        if self.relocation_model == RelocModel::Pic && self.os != "uefi" {
            assert!(self.dynamic_linking || self.position_independent_executables);
        }
        if self.static_position_independent_executables {
            assert!(self.position_independent_executables);
        }
        if self.position_independent_executables {
            assert!(self.executables);
        }

        // Check crt static stuff
        if self.crt_static_default || self.crt_static_allows_dylibs {
            assert!(self.crt_static_respected);
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
