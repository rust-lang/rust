use crate::session::Session;
use rustc_data_structures::profiling::VerboseTimingGuard;

impl Session {
    pub fn timer<'a>(&'a self, what: &'static str) -> VerboseTimingGuard<'a> {
        self.prof.verbose_generic_activity(what)
    }
    pub fn time<R>(&self, what: &'static str, f: impl FnOnce() -> R) -> R {
        self.prof.verbose_generic_activity(what).run(f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub enum NativeLibKind {
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC) included
    /// when linking a final binary, but not when archiving an rlib.
    StaticNoBundle,
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC) included
    /// when linking a final binary, but also included when archiving an rlib.
    StaticBundle,
    /// Dynamic library (e.g. `libfoo.so` on Linux)
    /// or an import library corresponding to a dynamic library (e.g. `foo.lib` on Windows/MSVC).
    Dylib,
    /// Dynamic library (e.g. `foo.dll` on Windows) without a corresponding import library.
    RawDylib,
    /// A macOS-specific kind of dynamic libraries.
    Framework,
    /// The library kind wasn't specified, `Dylib` is currently used as a default.
    Unspecified,
}

rustc_data_structures::impl_stable_hash_via_hash!(NativeLibKind);
