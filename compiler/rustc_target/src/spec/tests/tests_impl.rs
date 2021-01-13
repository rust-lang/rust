use super::super::*;

// Test target self-consistency and JSON encoding/decoding roundtrip.
pub(super) fn test_target(target: Target) {
    target.check_consistency();
    assert_eq!(Target::from_json(target.to_json()), Ok(target));
}

impl Target {
    fn check_consistency(&self) {
        assert!(self.is_like_windows || !self.is_like_msvc);
        // Check that LLD with the given flavor is treated identically to the linker it emulates.
        // If your target really needs to deviate from the rules below, except it and document the
        // reasons.
        assert_eq!(
            self.linker_flavor == LinkerFlavor::Msvc
                || self.linker_flavor == LinkerFlavor::Lld(LldFlavor::Link),
            self.lld_flavor == LldFlavor::Link,
        );
        assert_eq!(self.is_like_msvc, self.lld_flavor == LldFlavor::Link);
        for args in &[
            &self.pre_link_args,
            &self.late_link_args,
            &self.late_link_args_dynamic,
            &self.late_link_args_static,
            &self.post_link_args,
        ] {
            assert_eq!(
                args.get(&LinkerFlavor::Msvc),
                args.get(&LinkerFlavor::Lld(LldFlavor::Link)),
            );
            if args.contains_key(&LinkerFlavor::Msvc) {
                assert_eq!(self.lld_flavor, LldFlavor::Link);
            }
        }
        assert!(
            (self.pre_link_objects_fallback.is_empty()
                && self.post_link_objects_fallback.is_empty())
                || self.crt_objects_fallback.is_some()
        );
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
            || (self.env == "sgx" && self.vendor == "fortanix")
    }
}
