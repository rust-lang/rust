//! Static files bundled with documentation output.
//!
//! All the static files are included here for centralized access in case anything other than the
//! HTML rendering code (say, the theme checker) needs to access one of these files.
//!
//! Note about types: CSS and JavaScript files are included as `&'static str` to allow for the
//! minifier to run on them. All other files are included as `&'static [u8]` so they can be
//! directly written to a `Write` handle.

/// The file contents of the main `rustdoc.css` file, responsible for the core layout of the page.
crate static RUSTDOC_CSS: &str = include_str!("static/rustdoc.css");

/// The file contents of `settings.css`, responsible for the items on the settings page.
crate static SETTINGS_CSS: &str = include_str!("static/settings.css");

/// The file contents of the `noscript.css` file, used in case JS isn't supported or is disabled.
crate static NOSCRIPT_CSS: &str = include_str!("static/noscript.css");

/// The file contents of `normalize.css`, included to even out standard elements between browser
/// implementations.
crate static NORMALIZE_CSS: &str = include_str!("static/normalize.css");

/// The file contents of `main.js`, which contains the core JavaScript used on documentation pages,
/// including search behavior and docblock folding, among others.
crate static MAIN_JS: &str = include_str!("static/main.js");

/// The file contents of `settings.js`, which contains the JavaScript used to handle the settings
/// page.
crate static SETTINGS_JS: &str = include_str!("static/settings.js");

/// The file contents of `storage.js`, which contains functionality related to browser Local
/// Storage, used to store documentation settings.
crate static STORAGE_JS: &str = include_str!("static/storage.js");

/// The file contents of `brush.svg`, the icon used for the theme-switch button.
crate static BRUSH_SVG: &[u8] = include_bytes!("static/brush.svg");

/// The file contents of `wheel.svg`, the icon used for the settings button.
crate static WHEEL_SVG: &[u8] = include_bytes!("static/wheel.svg");

/// The file contents of `down-arrow.svg`, the icon used for the crate choice combobox.
crate static DOWN_ARROW_SVG: &[u8] = include_bytes!("static/down-arrow.svg");

/// The contents of `COPYRIGHT.txt`, the license listing for files distributed with documentation
/// output.
crate static COPYRIGHT: &[u8] = include_bytes!("static/COPYRIGHT.txt");

/// The contents of `LICENSE-APACHE.txt`, the text of the Apache License, version 2.0.
crate static LICENSE_APACHE: &[u8] = include_bytes!("static/LICENSE-APACHE.txt");

/// The contents of `LICENSE-MIT.txt`, the text of the MIT License.
crate static LICENSE_MIT: &[u8] = include_bytes!("static/LICENSE-MIT.txt");

/// The contents of `rust-logo.png`, the default icon of the documentation.
crate static RUST_LOGO: &[u8] = include_bytes!("static/rust-logo.png");
/// The default documentation favicons (SVG and PNG fallbacks)
crate static RUST_FAVICON_SVG: &[u8] = include_bytes!("static/favicon.svg");
crate static RUST_FAVICON_PNG_16: &[u8] = include_bytes!("static/favicon-16x16.png");
crate static RUST_FAVICON_PNG_32: &[u8] = include_bytes!("static/favicon-32x32.png");

/// The built-in themes given to every documentation site.
crate mod themes {
    /// The "light" theme, selected by default when no setting is available. Used as the basis for
    /// the `--check-theme` functionality.
    crate static LIGHT: &str = include_str!("static/themes/light.css");

    /// The "dark" theme.
    crate static DARK: &str = include_str!("static/themes/dark.css");

    /// The "ayu" theme.
    crate static AYU: &str = include_str!("static/themes/ayu.css");
}

/// Files related to the Fira Sans font.
crate mod fira_sans {
    /// The file `FiraSans-Regular.woff`, the Regular variant of the Fira Sans font.
    crate static REGULAR: &[u8] = include_bytes!("static/FiraSans-Regular.woff");

    /// The file `FiraSans-Medium.woff`, the Medium variant of the Fira Sans font.
    crate static MEDIUM: &[u8] = include_bytes!("static/FiraSans-Medium.woff");

    /// The file `FiraSans-LICENSE.txt`, the license text for the Fira Sans font.
    crate static LICENSE: &[u8] = include_bytes!("static/FiraSans-LICENSE.txt");
}

/// Files related to the Source Serif Pro font.
crate mod source_serif_pro {
    /// The file `SourceSerifPro-Regular.ttf.woff`, the Regular variant of the Source Serif Pro
    /// font.
    crate static REGULAR: &[u8] = include_bytes!("static/SourceSerifPro-Regular.ttf.woff");

    /// The file `SourceSerifPro-Bold.ttf.woff`, the Bold variant of the Source Serif Pro font.
    crate static BOLD: &[u8] = include_bytes!("static/SourceSerifPro-Bold.ttf.woff");

    /// The file `SourceSerifPro-It.ttf.woff`, the Italic variant of the Source Serif Pro font.
    crate static ITALIC: &[u8] = include_bytes!("static/SourceSerifPro-It.ttf.woff");

    /// The file `SourceSerifPro-LICENSE.txt`, the license text for the Source Serif Pro font.
    crate static LICENSE: &[u8] = include_bytes!("static/SourceSerifPro-LICENSE.md");
}

/// Files related to the Source Code Pro font.
crate mod source_code_pro {
    /// The file `SourceCodePro-Regular.woff`, the Regular variant of the Source Code Pro font.
    crate static REGULAR: &[u8] = include_bytes!("static/SourceCodePro-Regular.woff");

    /// The file `SourceCodePro-Semibold.woff`, the Semibold variant of the Source Code Pro font.
    crate static SEMIBOLD: &[u8] = include_bytes!("static/SourceCodePro-Semibold.woff");

    /// The file `SourceCodePro-LICENSE.txt`, the license text of the Source Code Pro font.
    crate static LICENSE: &[u8] = include_bytes!("static/SourceCodePro-LICENSE.txt");
}

/// Files related to the sidebar in rustdoc sources.
crate mod sidebar {
    /// File script to handle sidebar.
    crate static SOURCE_SCRIPT: &str = include_str!("static/source-script.js");
}
