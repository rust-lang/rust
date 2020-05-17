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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub enum NativeLibKind {
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC) included
    /// when linking a final binary, but not when archiving an rlib.
    StaticNoBundle,
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC) included
    /// when linking a final binary, but also included when archiving an rlib.
    StaticBundle,
    /// Windows dynamic library (`foo.dll`) without a corresponding import library.
    RawDylib,
    /// A macOS-specific kind of dynamic libraries.
    Framework,
    /// The library kind wasn't specified, dynamic linking is currently preferred.
    Unspecified,
}

rustc_data_structures::impl_stable_hash_via_hash!(NativeLibKind);
