//! Unified crate for re-exporting Unicode crates for the compiler.
//! Used to avoid having multiple different Unicode versions in use.

// If this changes, update https://github.com/rust-lang/reference/blob/HEAD/src/identifiers.md.
// At time of writing, updating this requires a T-lang FCP.
const UNICODE_VERSION: (u8, u8, u8) = (18, 0, 0);

const _: () = {
    let (major, minor, patch) = UNICODE_VERSION;
    let (major_, minor_, patch_) = char::UNICODE_VERSION;
    if major != major_ || minor != minor_ || patch != patch_ {
        // For now, upgrades for the Unicode version used by core and the compiler are independent.
        // Until this changes, the Unicode version in this crate and `char` may diverge by
        // 1 major version.
        // Update this const to make this explicit.
        const VERSIONS_DIFFER: bool = false;
        if !VERSIONS_DIFFER || major + 1 != major_ {
            panic!("rustc_unicode::UNICODE_VERSION does not match char::UNICODE_VERSION",)
        }
    }
};

macro_rules! reexport_and_check {
    ($($krate:ident),+ $(,)?) => {
        $(
            pub use $krate;
            const _: () = {
                let (major, minor, patch) = UNICODE_VERSION;
                let (major_, minor_, patch_) = $krate::UNICODE_VERSION;
                if (
                    major as u64 != major_ as u64 ||
                    minor as u64 != minor_ as u64 ||
                    patch as u64 != patch_ as u64
                ) {
                    panic!(
                        concat!(
                            stringify!($krate),
                            " does not have the correct Unicode version (`rustc_unicode::UNICODE_VERSION`)"
                        )
                    )
                }
            };
        )+
    }
}

reexport_and_check!(
    // tidy-alphabetical-start
    unicode_ident,
    unicode_normalization,
    unicode_properties,
    unicode_script,
    unicode_security,
    unicode_width,
    // tidy-alphabetical-end
);
