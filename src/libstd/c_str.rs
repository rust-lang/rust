// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast;
use iterator::Iterator;
use libc;
use ops::Drop;
use option::{Option, Some, None};
use ptr::RawPtr;
use ptr;
use str::StrSlice;
use vec::ImmutableVector;

/**
 * The representation of a C String.
 *
 * This structure wraps a `*libc::c_char`, and will automatically free the
 * memory it is pointing to when it goes out of scope.
 */
pub struct CString {
    priv buf: *libc::c_char,
}

impl<'self> CString {
    /**
     * Create a C String from a str.
     */
    pub fn from_str(s: &str) -> CString {
        s.to_c_str()
    }

    /**
     * Take the wrapped `*libc::c_char` from the `CString` wrapper.
     *
     * # Failure
     *
     * If the wrapper is empty.
     */
    pub unsafe fn take(&mut self) -> *libc::c_char {
        if self.buf.is_null() {
            fail!("CString has no wrapped `*libc::c_char`");
        }
        let buf = self.buf;
        self.buf = ptr::null();
        buf
    }

    /**
     * Puts a `*libc::c_char` into the `CString` wrapper.
     *
     * # Failure
     *
     * If the `*libc::c_char` is null.
     * If the wrapper is not empty.
     */
    pub fn put_back(&mut self, buf: *libc::c_char) {
        if buf.is_null() {
            fail!("attempt to put a null pointer into a CString");
        }
        if self.buf.is_not_null() {
            fail!("CString already wraps a `*libc::c_char`");
        }
        self.buf = buf;
    }

    /**
     * Calls a closure with a reference to the underlying `*libc::c_char`.
     */
    pub fn with_ref<T>(&self, f: &fn(*libc::c_char) -> T) -> T {
        if self.buf.is_null() {
            fail!("CString already wraps a `*libc::c_char`");
        }
        f(self.buf)
    }

    /**
     * Calls a closure with a mutable reference to the underlying `*libc::c_char`.
     */
    pub fn with_mut_ref<T>(&mut self, f: &fn(*mut libc::c_char) -> T) -> T {
        if self.buf.is_not_null() {
            fail!("CString already wraps a `*libc::c_char`");
        }
        f(unsafe { cast::transmute(self.buf) })
    }

    /**
     * Returns true if the CString does not wrap a `*libc::c_char`.
     */
    pub fn is_empty(&self) -> bool {
        self.buf.is_null()
    }

    /**
     * Returns true if the CString wraps a `*libc::c_char`.
     */
    pub fn is_not_empty(&self) -> bool {
        self.buf.is_not_null()
    }

    /**
     * Converts the CString into a `&[u8]` without copying.
     */
    pub fn as_bytes(&self) -> &'self [u8] {
        unsafe {
            let len = libc::strlen(self.buf) as uint;
            cast::transmute((self.buf, len + 1))
        }
    }

    /**
     * Return a CString iterator.
     */
    fn iter(&self) -> CStringIterator<'self> {
        CStringIterator {
            ptr: self.buf,
            lifetime: unsafe { cast::transmute(self.buf) },
        }
    }
}

impl Drop for CString {
    fn drop(&self) {
        if self.buf.is_not_null() {
            unsafe {
                libc::free(self.buf as *libc::c_void)
            };
        }
    }
}

/**
 * A generic trait for converting a value to a CString.
 */
pub trait ToCStr {
    /**
     * Create a C String.
     */
    fn to_c_str(&self) -> CString;
}

impl<'self> ToCStr for &'self str {
    /**
     * Create a C String from a `&str`.
     */
    fn to_c_str(&self) -> CString {
        self.as_bytes().to_c_str()
    }
}

impl<'self> ToCStr for &'self [u8] {
    /**
     * Create a C String from a `&[u8]`.
     */
    fn to_c_str(&self) -> CString {
        do self.as_imm_buf |self_buf, self_len| {
            unsafe {
                let buf = libc::malloc(self_len as u64 + 1) as *mut u8;
                if buf.is_null() {
                    fail!("failed to allocate memory!");
                }

                ptr::copy_memory(buf, self_buf, self_len);
                *ptr::mut_offset(buf, self_len as int) = 0;
                CString { buf: buf as *libc::c_char }
            }
        }
    }
}

/**
 * External iterator for a CString's bytes.
 *
 * Use with the `std::iterator` module.
 */
pub struct CStringIterator<'self> {
    priv ptr: *libc::c_char,
    priv lifetime: &'self libc::c_char, // FIXME: #5922
}

impl<'self> Iterator<libc::c_char> for CStringIterator<'self> {
    /**
     * Advance the iterator.
     */
    fn next(&mut self) -> Option<libc::c_char> {
        if self.ptr.is_null() {
            None
        } else {
            let ch = unsafe { *self.ptr };
            self.ptr = ptr::offset(self.ptr, 1);
            Some(ch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libc;
    use ptr;

    #[test]
    fn test_to_c_str() {
        do "".to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 0);
            }
        }

        do "hello".to_c_str().with_ref |buf| {
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'h' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 1), 'e' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 2), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 3), 'l' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 4), 'o' as libc::c_char);
                assert_eq!(*ptr::offset(buf, 5), 0);
            }
        }
    }

    #[test]
    fn test_take() {
        let mut c_str = "hello".to_c_str();
        unsafe { libc::free(c_str.take() as *libc::c_void) }
        assert!(c_str.is_empty());
    }

    #[test]
    fn test_take_and_put_back() {
        let mut c_str = "hello".to_c_str();
        assert!(c_str.is_not_empty());

        let buf = unsafe { c_str.take() };

        assert!(c_str.is_empty());

        c_str.put_back(buf);

        assert!(c_str.is_not_empty());
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_take_empty_fail() {
        let mut c_str = "hello".to_c_str();
        unsafe {
            libc::free(c_str.take() as *libc::c_void);
            c_str.take();
        }
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_put_back_null_fail() {
        let mut c_str = "hello".to_c_str();
        c_str.put_back(ptr::null());
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_put_back_full_fail() {
        let mut c_str = "hello".to_c_str();
        c_str.put_back(0xdeadbeef as *libc::c_char);
    }

    fn test_with() {
        let c_str = "hello".to_c_str();
        let len = unsafe { c_str.with_ref(|buf| libc::strlen(buf)) };
        assert!(c_str.is_not_empty());
        assert_eq!(len, 5);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_with_empty_fail() {
        let mut c_str = "hello".to_c_str();
        unsafe { libc::free(c_str.take() as *libc::c_void) }
        c_str.with_ref(|_| ());
    }
}
