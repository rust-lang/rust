// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// To avoid problems with Windows UAC installer detection heuristics,
// Rust-produced executables need an application manifest.
// For details, see issue #10512.

// No-op on other platforms.

use driver::session::Session;
use std::path::Path;

#[cfg(not(windows))]
pub fn postprocess_executable(_sess: Session, _filename: &Path) {}

#[cfg(windows)]
pub fn postprocess_executable(sess: Session, filename: &Path) {

    let default_manifest = concat!(
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
        "<assembly xmlns='urn:schemas-microsoft-com:asm.v1' manifestVersion='1.0'>",
        "  <trustInfo xmlns='urn:schemas-microsoft-com:asm.v3'>",
        "    <security>",
        "      <requestedPrivileges>",
        "        <requestedExecutionLevel level='asInvoker' uiAccess='false' />",
        "      </requestedPrivileges>",
        "    </security>",
        "  </trustInfo>",
        "</assembly>");

    match windows::embed_manifest(filename, default_manifest) {
        Ok(_) => (),
        Err(msg) => sess.err(format!("Could not embed application manifest: {}", msg))
    }
}

#[cfg(windows)]
mod windows {
    use std::libc::types::os::arch::extra::{BOOL,WORD,DWORD,HANDLE,LPCWSTR,LPCVOID};
    use std::libc::consts::os::extra::FALSE;
    use std::cast::transmute;
    use std::os;

    // FIXME #9053: should import as_utf16_p from std rather than re-defining here
    //use std::os::win32::as_utf16_p;
    fn as_utf16_p<T>(s: &str, f: |*u16| -> T) -> T {
        let mut t = s.to_utf16();
        // Null terminate before passing on.
        t.push(0u16);
        f(t.as_ptr())
    }

    #[link_name = "kernel32"]
    extern "system" {
        pub fn BeginUpdateResourceW(pFileName: LPCWSTR,
                                    bDeleteExistingResources: BOOL) -> HANDLE;
        pub fn UpdateResourceW(hUpdate: HANDLE,
                               lpType: LPCWSTR,
                               lpName: LPCWSTR,
                               wLanguage: WORD,
                               lpData: LPCVOID,
                               cbData: DWORD) -> BOOL;
        pub fn EndUpdateResourceW(hUpdate: HANDLE,
                                  fDiscard: BOOL) -> BOOL;
    }

    fn MAKEINTRESOURCEW(id: int) -> LPCWSTR {
        unsafe{ transmute(id) }
    }

    pub fn embed_manifest(filename: &Path,
                          manifest: &str) -> Result<(),~str> {
        unsafe {
            let hUpdate = as_utf16_p(filename.as_str().unwrap(), |path| {
                BeginUpdateResourceW(path, FALSE)
            });
            if hUpdate.is_null() {
                return Err(format!("failure in BeginUpdateResourceW: {}", os::last_os_error()));
            }

            let ok = UpdateResourceW(hUpdate,
                                     MAKEINTRESOURCEW(24), // RT_MANIFEST
                                     MAKEINTRESOURCEW(1),  // CREATEPROCESS_MANIFEST_RESOURCE_ID
                                     0,                    // LANG_NEUTRAL, SUBLANG_NEUTRAL
                                     manifest.as_ptr() as LPCVOID,
                                     manifest.len() as u32);
            if ok == FALSE {
                return Err(format!("failure in UpdateResourceW: {}", os::last_os_error()));
            }

            let ok = EndUpdateResourceW(hUpdate, FALSE);
            if ok == FALSE {
                return Err(format!("failure in EndUpdateResourceW: {}", os::last_os_error()));
            }
            Ok(())
        }
    }
}
