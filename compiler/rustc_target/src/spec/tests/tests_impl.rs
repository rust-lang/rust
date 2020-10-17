use super::super::*;

// Test target self-consistency and JSON encoding/decoding roundtrip.
pub(super) fn test_target(target: Target) {
    target.check_consistency();
    assert_eq!(Target::from_json(target.to_json()), Ok(target));
}

impl Target {
    fn check_consistency(&self) {
        // Check that LLD with the given flavor is treated identically to the linker it emulates.
        // If your target really needs to deviate from the rules below, except it and document the
        // reasons.
        assert_eq!(
            self.linker_flavor == LinkerFlavor::Msvc
                || self.linker_flavor == LinkerFlavor::Lld(LldFlavor::Link),
            self.options.lld_flavor == LldFlavor::Link,
        );
        for args in &[
            &self.options.pre_link_args,
            &self.options.late_link_args,
            &self.options.late_link_args_dynamic,
            &self.options.late_link_args_static,
            &self.options.post_link_args,
        ] {
            assert_eq!(
                args.get(&LinkerFlavor::Msvc),
                args.get(&LinkerFlavor::Lld(LldFlavor::Link)),
            );
            if args.contains_key(&LinkerFlavor::Msvc) {
                assert_eq!(self.options.lld_flavor, LldFlavor::Link);
            }
        }
        assert!(
            (self.options.pre_link_objects_fallback.is_empty()
                && self.options.post_link_objects_fallback.is_empty())
                || self.options.crt_objects_fallback.is_some()
        );
    }
}
