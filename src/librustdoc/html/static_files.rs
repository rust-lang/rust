//! Static files bundled with documentation output.
//!
//! All the static files are included here for centralized access in case anything other than the
//! HTML rendering code (say, the theme checker) needs to access one of these files.

use std::path::{Path, PathBuf};
use std::{fmt, str};

pub(crate) struct StaticFile {
    pub(crate) filename: PathBuf,
    pub(crate) src_bytes: &'static [u8],
    pub(crate) minified_bytes: &'static [u8],
}

impl StaticFile {
    fn new(
        filename: &str,
        src_bytes: &'static [u8],
        minified_bytes: &'static [u8],
        sha256: &'static str,
    ) -> StaticFile {
        Self { filename: static_filename(filename, sha256), src_bytes, minified_bytes }
    }

    pub(crate) fn output_filename(&self) -> &Path {
        &self.filename
    }
}

/// The Display implementation for a StaticFile outputs its filename. This makes it
/// convenient to interpolate static files into HTML templates.
impl fmt::Display for StaticFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.output_filename().display())
    }
}

/// Insert the provided suffix into a filename just before the extension.
pub(crate) fn suffix_path(filename: &str, suffix: &str) -> PathBuf {
    // We use splitn vs Path::extension here because we might get a filename
    // like `style.min.css` and we want to process that into
    // `style-suffix.min.css`.  Path::extension would just return `css`
    // which would result in `style.min-suffix.css` which isn't what we
    // want.
    let (base, ext) = filename.split_once('.').unwrap();
    let filename = format!("{base}{suffix}.{ext}");
    filename.into()
}

pub(crate) fn static_filename(filename: &str, sha256: &str) -> PathBuf {
    let filename = filename.rsplit('/').next().unwrap();
    suffix_path(filename, sha256)
}

macro_rules! static_files {
    ($($field:ident => $file_path:literal,)+) => {
        pub(crate) struct StaticFiles {
            $(pub $field: StaticFile,)+
        }

        // sha256 files are generated in build.rs
        pub(crate) static STATIC_FILES: std::sync::LazyLock<StaticFiles> = std::sync::LazyLock::new(|| StaticFiles {
            $($field: StaticFile::new($file_path, include_bytes!($file_path), include_bytes!(concat!(env!("OUT_DIR"), "/", $file_path, ".min")), include_str!(concat!(env!("OUT_DIR"), "/", $file_path, ".sha256"))),)+
        });

        pub(crate) fn for_each<E>(f: impl Fn(&StaticFile) -> Result<(), E>) -> Result<(), E> {
            for sf in [
            $(&STATIC_FILES.$field,)+
            ] {
                f(sf)?
            }
            Ok(())
        }
    }
}

static_files! {
    rustdoc_css => "static/css/rustdoc.css",
    noscript_css => "static/css/noscript.css",
    normalize_css => "static/css/normalize.css",
    main_js => "static/js/main.js",
    search_js => "static/js/search.js",
    stringdex_js => "static/js/stringdex.js",
    settings_js => "static/js/settings.js",
    src_script_js => "static/js/src-script.js",
    storage_js => "static/js/storage.js",
    scrape_examples_js => "static/js/scrape-examples.js",
    copyright => "static/COPYRIGHT.txt",
    license_apache => "static/LICENSE-APACHE.txt",
    license_mit => "static/LICENSE-MIT.txt",
    rust_logo_svg => "static/images/rust-logo.svg",
    rust_favicon_svg => "static/images/favicon.svg",
    rust_favicon_png_32 => "static/images/favicon-32x32.png",
    fira_sans_italic => "static/fonts/FiraSans-Italic.woff2",
    fira_sans_regular => "static/fonts/FiraSans-Regular.woff2",
    fira_sans_medium => "static/fonts/FiraSans-Medium.woff2",
    fira_sans_medium_italic => "static/fonts/FiraSans-MediumItalic.woff2",
    fira_mono_regular => "static/fonts/FiraMono-Regular.woff2",
    fira_mono_medium => "static/fonts/FiraMono-Medium.woff2",
    fira_sans_license => "static/fonts/FiraSans-LICENSE.txt",
    source_serif_4_regular => "static/fonts/SourceSerif4-Regular.ttf.woff2",
    source_serif_4_semibold => "static/fonts/SourceSerif4-Semibold.ttf.woff2",
    source_serif_4_bold => "static/fonts/SourceSerif4-Bold.ttf.woff2",
    source_serif_4_italic => "static/fonts/SourceSerif4-It.ttf.woff2",
    source_serif_4_license => "static/fonts/SourceSerif4-LICENSE.md",
    source_code_pro_regular => "static/fonts/SourceCodePro-Regular.ttf.woff2",
    source_code_pro_semibold => "static/fonts/SourceCodePro-Semibold.ttf.woff2",
    source_code_pro_italic => "static/fonts/SourceCodePro-It.ttf.woff2",
    source_code_pro_license => "static/fonts/SourceCodePro-LICENSE.txt",
    nanum_barun_gothic_regular => "static/fonts/NanumBarunGothic.ttf.woff2",
    nanum_barun_gothic_license => "static/fonts/NanumBarunGothic-LICENSE.txt",
}

pub(crate) static SCRAPE_EXAMPLES_HELP_MD: &str = include_str!("static/scrape-examples-help.md");
