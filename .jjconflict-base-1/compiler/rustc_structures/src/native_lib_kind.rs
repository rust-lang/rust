#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(StableHash, Encodable_NoContext, Decodable_NoContext))]
pub enum NativeLibKind {
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC)
    Static {
        /// Whether to bundle objects from static library into produced rlib
        bundle: Option<bool>,
        /// Whether to link static library without throwing any object files away
        whole_archive: Option<bool>,
        /// Whether to export c static library symbols
        export_symbols: Option<bool>,
    },
    /// Dynamic library (e.g. `libfoo.so` on Linux)
    /// or an import library corresponding to a dynamic library (e.g. `foo.lib` on Windows/MSVC).
    Dylib {
        /// Whether the dynamic library will be linked only if it satisfies some undefined symbols
        as_needed: Option<bool>,
    },
    /// Dynamic library (e.g. `foo.dll` on Windows) without a corresponding import library.
    /// On Linux, it refers to a generated shared library stub.
    RawDylib {
        /// Whether the dynamic library will be linked only if it satisfies some undefined symbols
        as_needed: Option<bool>,
    },
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
            NativeLibKind::Static { bundle, whole_archive, export_symbols } => {
                bundle.is_some() || whole_archive.is_some() || export_symbols.is_some()
            }
            NativeLibKind::Dylib { as_needed }
            | NativeLibKind::Framework { as_needed }
            | NativeLibKind::RawDylib { as_needed } => as_needed.is_some(),
            NativeLibKind::Unspecified
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
            NativeLibKind::Dylib { .. }
                | NativeLibKind::RawDylib { .. }
                | NativeLibKind::Unspecified
        )
    }
}
