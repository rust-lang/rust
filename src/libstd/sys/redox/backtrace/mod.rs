// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// See sys/unix/backtrace/mod.rs for an explanation of the method used here.

pub use self::tracing::unwind_backtrace;
pub use self::printing::{foreach_symbol_fileline, resolve_symname};

// tracing impls:
mod tracing;
// symbol resolvers:
mod printing;

pub mod gnu {
    use io;
    use fs;
    use libc::c_char;
    use vec::Vec;
    use ffi::OsStr;
    use os::unix::ffi::OsStrExt;
    use io::Read;

    pub fn get_executable_filename() -> io::Result<(Vec<c_char>, fs::File)> {
        let mut exefile = fs::File::open("sys:exe")?;
        let mut exename = Vec::new();
        exefile.read_to_end(&mut exename)?;
        if exename.last() == Some(&b'\n') {
            exename.pop();
        }
        let file = fs::File::open(OsStr::from_bytes(&exename))?;
        Ok((exename.into_iter().map(|c| c as c_char).collect(), file))
    }
}

pub struct BacktraceContext;
