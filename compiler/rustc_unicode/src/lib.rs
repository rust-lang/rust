//! Unified crate for re-exporting Unicode crates for the compiler.
//! Used to avoid having multiple different Unicode versions in use.

// If this changes, update https://github.com/rust-lang/reference/blob/HEAD/src/identifiers.md.
const UNICODE_VERSION: (u8, u8, u8) = (17, 0, 0);

macro_rules! reexport_and_check {
    ($($krate:ident),+ $(,)?) => {
        $(
            pub use $krate;
            const _: () = {
                let (major, minor, patch) = crate::UNICODE_VERSION;
                let (major_, minor_, patch_) = $krate::UNICODE_VERSION;
                if major != major_ as u8 || minor != minor_ as u8 || patch != patch_ as u8 {
                    panic!(
                        concat!(
                            stringify!($krate),
                            " does not have the correct Unicode version (`rustc-unicode::UNICODE_VERSION`)"
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
    // FIXME: Update to version supporting Unicode 17
    // unicode_script,
    // unicode_security,
    unicode_width,
    // tidy-alphabetical-end
);
pub use unicode_script;
pub use unicode_security;
