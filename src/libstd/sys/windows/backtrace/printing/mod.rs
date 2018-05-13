// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(target_env = "msvc")]
#[path = "msvc.rs"]
mod printing;

#[cfg(target_env = "gnu")]
mod printing {
    pub use sys_common::gnu::libbacktrace::{foreach_symbol_fileline, resolve_symname};

    // dummy functions to mirror those present in msvc version.
    use sys::dynamic_lib::DynamicLibrary;
    use io;
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
