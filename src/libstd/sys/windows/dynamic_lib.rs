use crate::os::windows::prelude::*;

use crate::ffi::{CString, OsStr};
use crate::io;
use crate::sys::c;

pub struct DynamicLibrary {
    handle: c::HMODULE,
}

impl DynamicLibrary {
    pub fn open(filename: &str) -> io::Result<DynamicLibrary> {
        let filename = OsStr::new(filename)
                             .encode_wide()
                             .chain(Some(0))
                             .collect::<Vec<_>>();
        let result = unsafe {
            c::LoadLibraryW(filename.as_ptr())
        };
        if result.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(DynamicLibrary { handle: result })
        }
    }

    pub fn symbol(&self, symbol: &str) -> io::Result<usize> {
        let symbol = CString::new(symbol)?;
        unsafe {
            match c::GetProcAddress(self.handle, symbol.as_ptr()) as usize {
                0 => Err(io::Error::last_os_error()),
                n => Ok(n),
            }
        }
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            c::FreeLibrary(self.handle);
        }
    }
}
