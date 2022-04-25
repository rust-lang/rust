use std::str::FromStr;

use crate::config::{LinkSelfContained, LinkSelfContainedCrt, LinkSelfContainedLinker};

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
