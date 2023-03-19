use crate::session::Session;
use rustc_data_structures::profiling::VerboseTimingGuard;
use std::path::{Path, PathBuf};

impl Session {
    pub fn timer(&self, what: &'static str) -> VerboseTimingGuard<'_> {
        self.prof.verbose_generic_activity(what)
    }
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
        Self { original: path.to_owned(), canonicalized: std::fs::canonicalize(path).ok() }
    }

    pub fn canonicalized(&self) -> &PathBuf {
        self.canonicalized.as_ref().unwrap_or(self.original())
    }

    pub fn original(&self) -> &PathBuf {
        &self.original
    }
}
