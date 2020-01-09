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
pub enum NativeLibraryKind {
    /// native static library (.a archive)
    NativeStatic,
    /// native static library, which doesn't get bundled into .rlibs
    NativeStaticNobundle,
    /// macOS-specific
    NativeFramework,
    /// Windows dynamic library without import library.
    NativeRawDylib,
    /// default way to specify a dynamic library
    NativeUnknown,
}

rustc_data_structures::impl_stable_hash_via_hash!(NativeLibraryKind);
