// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// GetLastError doesn't seem to work with stack switching

#[cfg(windows)]
mod kernel32 {
  extern "system" {
    pub fn SetLastError(err: uint);
    pub fn GetLastError() -> uint;
  }
}


#[cfg(windows)]
pub fn main() {
    unsafe {
        let expected = 1234u;
        kernel32::SetLastError(expected);
        let actual = kernel32::GetLastError();
        info!("actual = {}", actual);
        assert_eq!(expected, actual);
    }
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
pub fn main() { }
