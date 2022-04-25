use std::str::FromStr;

use crate::config::*;
use rustc_target::spec::{LinkerFlavor, LldFlavor};

/// When adding support for `-C link-self-contained=linker`, we want to ensure the existing
/// values are still supported as-is: they are option values that can be used on stable.
#[test]
pub fn parse_stable_self_contained() {
    // The existing default is when the argument is not used, the behavior depends on the
    // target.
    assert_eq!(
        LinkSelfContained::default(),
        LinkSelfContained { crt: LinkSelfContainedCrt::Auto, linker: LinkSelfContainedLinker::Off }
    );

    // Turning the flag `on` should currently only enable the default on stable.
    assert_eq!(
        LinkSelfContained::from_str("on"),
        Ok(LinkSelfContained {
            crt: LinkSelfContainedCrt::On,
            linker: LinkSelfContainedLinker::Off
        })
    );

    // Turning the flag `off` applies to both facets.
    assert_eq!(
        LinkSelfContained::from_str("off"),
        Ok(LinkSelfContained {
            crt: LinkSelfContainedCrt::Off,
            linker: LinkSelfContainedLinker::Off
        })
    );

    assert_eq!(
        LinkSelfContained::from_str("crt"),
        Ok(LinkSelfContained {
            crt: LinkSelfContainedCrt::On,
            linker: LinkSelfContainedLinker::Off
        })
    );
}

#[test]
pub fn parse_self_contained_with_linker() {
    // Turning the linker on doesn't change the CRT behavior
    assert_eq!(
        LinkSelfContained::from_str("linker"),
        Ok(LinkSelfContained {
            crt: LinkSelfContainedCrt::Auto,
            linker: LinkSelfContainedLinker::On
        })
    );

    assert_eq!(
        LinkSelfContained::from_str("all"),
        Ok(LinkSelfContained {
            crt: LinkSelfContainedCrt::On,
            linker: LinkSelfContainedLinker::On
        })
    );

    // If `linker` is turned on by default someday, we need to be able to go back to the current
    // default.
    assert_eq!(
        LinkSelfContained::from_str("auto"),
        Ok(LinkSelfContained {
            crt: LinkSelfContainedCrt::Auto,
            linker: LinkSelfContainedLinker::Off
        })
    );
}

/// When adding enrichments to `-C linker-flavor`, we want to ensure the existing `rustc_target`
/// `LinkerFlavor`s are still supported as-is: they are option values that can be used on
/// stable.
#[test]
pub fn parse_well_known_linker_flavor() {
    // All `LinkerFlavor`s are wrapped as a whole, so there's no particular need to be
    // exhaustive here.
    assert_eq!(LinkerFlavorCli::from_str("gcc"), Ok(LinkerFlavorCli::WellKnown(LinkerFlavor::Gcc)));
    assert_eq!(
        LinkerFlavorCli::from_str("msvc"),
        Ok(LinkerFlavorCli::WellKnown(LinkerFlavor::Msvc))
    );
    assert_eq!(
        LinkerFlavorCli::from_str("bpf-linker"),
        Ok(LinkerFlavorCli::WellKnown(LinkerFlavor::BpfLinker))
    );
    assert_eq!(
        LinkerFlavorCli::from_str("lld-link"),
        Ok(LinkerFlavorCli::WellKnown(LinkerFlavor::Lld(LldFlavor::Link)))
    );
    assert_eq!(
        LinkerFlavorCli::from_str("ld64.lld"),
        Ok(LinkerFlavorCli::WellKnown(LinkerFlavor::Lld(LldFlavor::Ld64)))
    );

    // While other invalid values for well-known flavors are already errors
    assert_eq!(LinkerFlavorCli::from_str("unknown_linker"), Err(()));
}

/// Enrichments can currently allow for the `gcc` flavor to specify for a given linker to be
/// used, much like you'd use `-fuse-ld` as a link arg. When using `-C
/// linker-flavor=gcc:$linker`, the `$linker` will be passed directly to `cc`.
#[test]
pub fn parse_gcc_enrichment_linker_flavor() {
    assert_eq!(
        LinkerFlavorCli::from_str("gcc:lld"),
        Ok(LinkerFlavorCli::Gcc { use_ld: "lld".to_string() })
    );
    assert_eq!(
        LinkerFlavorCli::from_str("gcc:gold"),
        Ok(LinkerFlavorCli::Gcc { use_ld: "gold".to_string() })
    );

    // No linker actually mentioned
    assert_eq!(LinkerFlavorCli::from_str("gcc:"), Err(()));

    // Only one `gcc:` separator allowed
    assert_eq!(LinkerFlavorCli::from_str("gcc:gcc:"), Err(()));
    assert_eq!(LinkerFlavorCli::from_str("gcc:gcc:linker"), Err(()));
}
