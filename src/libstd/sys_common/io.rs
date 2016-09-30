// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use io;
use io::ErrorKind;
use io::Read;
use slice::from_raw_parts_mut;

pub const DEFAULT_BUF_SIZE: usize = 8 * 1024;

// Provides read_to_end functionality over an uninitialized buffer.
// This function is unsafe because it calls the underlying
// read function with a slice into uninitialized memory. The default
// implementation of read_to_end for readers will zero out new memory in
// the buf before passing it to read, but avoiding this zero can often
// lead to a fairly significant performance win.
//
// Implementations using this method have to adhere to two guarantees:
//  *  The implementation of read never reads the buffer provided.
//  *  The implementation of read correctly reports how many bytes were written.
pub unsafe fn read_to_end_uninitialized(r: &mut Read, buf: &mut Vec<u8>) -> io::Result<usize> {

    let start_len = buf.len();
    buf.reserve(16);

    // Always try to read into the empty space of the vector (from the length to the capacity).
    // If the vector ever fills up then we reserve an extra byte which should trigger the normal
    // reallocation routines for the vector, which will likely double the size.
    //
    // This function is similar to the read_to_end function in std::io, but the logic about
    // reservations and slicing is different enough that this is duplicated here.
    loop {
        if buf.len() == buf.capacity() {
            buf.reserve(1);
        }

        let buf_slice = from_raw_parts_mut(buf.as_mut_ptr().offset(buf.len() as isize),
                                           buf.capacity() - buf.len());

        match r.read(buf_slice) {
            Ok(0) => { return Ok(buf.len() - start_len); }
            Ok(n) => { let len = buf.len() + n; buf.set_len(len); },
            Err(ref e) if e.kind() == ErrorKind::Interrupted => { }
            Err(e) => { return Err(e); }
        }
    }
}

#[cfg(test)]
#[allow(dead_code)] // not used on emscripten
pub mod test {
    use path::{Path, PathBuf};
    use env;
    use rand::{self, Rng};
    use fs;

    pub struct TempDir(PathBuf);

    impl TempDir {
        pub fn join(&self, path: &str) -> PathBuf {
            let TempDir(ref p) = *self;
            p.join(path)
        }

        pub fn path<'a>(&'a self) -> &'a Path {
            let TempDir(ref p) = *self;
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Gee, seeing how we're testing the fs module I sure hope that we
            // at least implement this correctly!
            let TempDir(ref p) = *self;
            fs::remove_dir_all(p).unwrap();
        }
    }

    pub fn tmpdir() -> TempDir {
        let p = env::temp_dir();
        let mut r = rand::thread_rng();
        let ret = p.join(&format!("rust-{}", r.next_u32()));
        fs::create_dir(&ret).unwrap();
        TempDir(ret)
    }
}

#[cfg(test)]
mod tests {
    use io::prelude::*;
    use super::*;
    use io;
    use io::{ErrorKind, Take, Repeat, repeat};
    use slice::from_raw_parts;

    struct ErrorRepeat {
        lr: Take<Repeat>
    }

    fn error_repeat(byte: u8, limit: u64) -> ErrorRepeat {
        ErrorRepeat { lr: repeat(byte).take(limit) }
    }

    impl Read for ErrorRepeat {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let ret = self.lr.read(buf);
            if let Ok(0) = ret {
                return Err(io::Error::new(ErrorKind::Other, ""))
            }
            ret
        }
    }

    fn init_vec_data() -> Vec<u8> {
        let mut vec = vec![10u8; 200];
        unsafe { vec.set_len(0); }
        vec
    }

    fn assert_all_eq(buf: &[u8], value: u8) {
        for n in buf {
            assert_eq!(*n, value);
        }
    }

    fn validate(buf: &Vec<u8>, good_read_len: usize) {
        assert_all_eq(buf, 1u8);
        let cap = buf.capacity();
        let end_slice = unsafe { from_raw_parts(buf.as_ptr().offset(good_read_len as isize),
                                                    cap - good_read_len) };
        assert_all_eq(end_slice, 10u8);
    }

    #[test]
    fn read_to_end_uninit_error() {
        let mut er = error_repeat(1,100);
        let mut vec = init_vec_data();
        if let Err(_) = unsafe { read_to_end_uninitialized(&mut er, &mut vec) } {
            validate(&vec, 100);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn read_to_end_uninit_zero_len_vec() {
        let mut er = repeat(1).take(100);
        let mut vec = Vec::new();
        let n = unsafe{ read_to_end_uninitialized(&mut er, &mut vec).unwrap() };
        assert_all_eq(&vec, 1u8);
        assert_eq!(vec.len(), n);
    }

    #[test]
    fn read_to_end_uninit_good() {
        let mut er = repeat(1).take(100);
        let mut vec = init_vec_data();
        let n = unsafe{ read_to_end_uninitialized(&mut er, &mut vec).unwrap() };
        validate(&vec, 100);
        assert_eq!(vec.len(), n);
    }

    #[bench]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn bench_uninitialized(b: &mut ::test::Bencher) {
        b.iter(|| {
            let mut lr = repeat(1).take(10000000);
            let mut vec = Vec::with_capacity(1024);
            unsafe { read_to_end_uninitialized(&mut lr, &mut vec) }
        });
    }
}
