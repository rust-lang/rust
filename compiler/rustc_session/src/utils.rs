use crate::session::Session;
use rustc_data_structures::profiling::VerboseTimingGuard;
use rustc_fs_util::try_canonicalize;
use std::path::{Path, PathBuf};

impl Session {
    pub fn timer(&self, what: &'static str) -> VerboseTimingGuard<'_> {
        self.prof.verbose_generic_activity(what)
    }
    /// Used by `-Z self-profile`.
    pub fn time<R>(&self, what: &'static str, f: impl FnOnce() -> R) -> R {
        self.prof.verbose_generic_activity(what).run(f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub enum NativeLibKind {
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC)
    Static {
        /// Whether to bundle objects from static library into produced rlib
        bundle: Option<bool>,
        /// Whether to link static library without throwing any object files away
        whole_archive: Option<bool>,
    },
    /// Dynamic library (e.g. `libfoo.so` on Linux)
    /// or an import library corresponding to a dynamic library (e.g. `foo.lib` on Windows/MSVC).
    Dylib {
        /// Whether the dynamic library will be linked only if it satisfies some undefined symbols
        as_needed: Option<bool>,
    },
    /// Dynamic library (e.g. `foo.dll` on Windows) without a corresponding import library.
    RawDylib,
    /// A macOS-specific kind of dynamic libraries.
    Framework {
        /// Whether the framework will be linked only if it satisfies some undefined symbols
        as_needed: Option<bool>,
    },
    /// Argument which is passed to linker, relative order with libraries and other arguments
    /// is preserved
    LinkArg,

    /// Module imported from WebAssembly
    WasmImportModule,

    /// The library kind wasn't specified, `Dylib` is currently used as a default.
    Unspecified,
}

impl NativeLibKind {
    pub fn has_modifiers(&self) -> bool {
        match self {
            NativeLibKind::Static { bundle, whole_archive } => {
                bundle.is_some() || whole_archive.is_some()
            }
            NativeLibKind::Dylib { as_needed } | NativeLibKind::Framework { as_needed } => {
                as_needed.is_some()
            }
            NativeLibKind::RawDylib
            | NativeLibKind::Unspecified
            | NativeLibKind::LinkArg
            | NativeLibKind::WasmImportModule => false,
        }
    }

    pub fn is_statically_included(&self) -> bool {
        matches!(self, NativeLibKind::Static { .. })
    }

    pub fn is_dllimport(&self) -> bool {
        matches!(
            self,
            NativeLibKind::Dylib { .. } | NativeLibKind::RawDylib | NativeLibKind::Unspecified
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub struct NativeLib {
    pub name: String,
    pub new_name: Option<String>,
    pub kind: NativeLibKind,
    pub verbatim: Option<bool>,
}

impl NativeLib {
    pub fn has_modifiers(&self) -> bool {
        self.verbatim.is_some() || self.kind.has_modifiers()
    }
}

/// A path that has been canonicalized along with its original, non-canonicalized form
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalizedPath {
    // Optional since canonicalization can sometimes fail
    canonicalized: Option<PathBuf>,
    original: PathBuf,
}

impl CanonicalizedPath {
    pub fn new(path: &Path) -> Self {
        Self { original: path.to_owned(), canonicalized: try_canonicalize(path).ok() }
    }

    pub fn canonicalized(&self) -> &PathBuf {
        self.canonicalized.as_ref().unwrap_or(self.original())
    }

    pub fn original(&self) -> &PathBuf {
        &self.original
    }
}

/// Gets a list of extra command-line flags provided by the user, as strings.
///
/// This function is used during ICEs to show more information useful for
/// debugging, since some ICEs only happens with non-default compiler flags
/// (and the users don't always report them).
pub fn extra_compiler_flags() -> Option<(Vec<String>, bool)> {
    const ICE_REPORT_COMPILER_FLAGS: &[&str] = &["-Z", "-C", "--crate-type"];

    const ICE_REPORT_COMPILER_FLAGS_EXCLUDE: &[&str] = &["metadata", "extra-filename"];

    const ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE: &[&str] = &["incremental"];

    let mut args = std::env::args_os().map(|arg| arg.to_string_lossy().to_string()).peekable();

    let mut result = Vec::new();
    let mut excluded_cargo_defaults = false;
    while let Some(arg) = args.next() {
        if let Some(a) = ICE_REPORT_COMPILER_FLAGS.iter().find(|a| arg.starts_with(*a)) {
            let content = if arg.len() == a.len() {
                // A space-separated option, like `-C incremental=foo` or `--crate-type rlib`
                match args.next() {
                    Some(arg) => arg.to_string(),
                    None => continue,
                }
            } else if arg.get(a.len()..a.len() + 1) == Some("=") {
                // An equals option, like `--crate-type=rlib`
                arg[a.len() + 1..].to_string()
            } else {
                // A non-space option, like `-Cincremental=foo`
                arg[a.len()..].to_string()
            };
            let option = content.split_once('=').map(|s| s.0).unwrap_or(&content);
            if ICE_REPORT_COMPILER_FLAGS_EXCLUDE.iter().any(|exc| option == *exc) {
                excluded_cargo_defaults = true;
            } else {
                result.push(a.to_string());
                match ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE.iter().find(|s| option == **s) {
                    Some(s) => result.push(format!("{s}=[REDACTED]")),
                    None => result.push(content),
                }
            }
        }
    }

    if !result.is_empty() { Some((result, excluded_cargo_defaults)) } else { None }
}

pub(crate) fn is_ascii_ident(string: &str) -> bool {
    let mut chars = string.chars();
    if let Some(start) = chars.next()
        && (start.is_ascii_alphabetic() || start == '_')
    {
        chars.all(|char| char.is_ascii_alphanumeric() || char == '_')
    } else {
        false
    }
}
