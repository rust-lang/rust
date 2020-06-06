use super::super::*;

pub(super) fn test_target(target: TargetResult) {
    // Grab the TargetResult struct. If we successfully retrieved
    // a Target, then the test JSON encoding/decoding can run for this
    // Target on this testing platform (i.e., checking the iOS targets
    // only on a Mac test platform).
    if let Ok(original) = target {
        original.check_consistency();
        let as_json = original.to_json();
        let parsed = Target::from_json(as_json).unwrap();
        assert_eq!(original, parsed);
    }
}

impl Target {
    fn check_consistency(&self) {
        // Check that LLD with the given flavor is treated identically to the linker it emulates.
        // If you target really needs to deviate from the rules below, whitelist it
        // and document the reasons.
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
