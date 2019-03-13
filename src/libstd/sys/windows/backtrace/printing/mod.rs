#[cfg(target_env = "msvc")]
#[path = "msvc.rs"]
mod printing;

#[cfg(target_env = "gnu")]
mod printing {
    pub use crate::sys_common::gnu::libbacktrace::{foreach_symbol_fileline, resolve_symname};

    // dummy functions to mirror those present in msvc version.
    use crate::sys::dynamic_lib::DynamicLibrary;
    use crate::io;
    pub struct PrintingFnsEx {}
    pub struct PrintingFns64 {}
    pub fn load_printing_fns_ex(_: &DynamicLibrary) -> io::Result<PrintingFnsEx> {
        Ok(PrintingFnsEx{})
    }
    pub fn load_printing_fns_64(_: &DynamicLibrary) -> io::Result<PrintingFns64> {
        Ok(PrintingFns64{})
    }
}

pub use self::printing::{foreach_symbol_fileline, resolve_symname};
pub use self::printing::{load_printing_fns_ex, load_printing_fns_64,
                         PrintingFnsEx, PrintingFns64};
