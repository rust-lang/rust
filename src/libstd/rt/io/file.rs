// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use super::support::PathLike;
use super::{Reader, Writer, Seek};
use super::SeekStyle;

/// # FIXME #7785
/// * Ugh, this is ridiculous. What is the best way to represent these options?
enum FileMode {
    /// Opens an existing file. IoError if file does not exist.
    Open,
    /// Creates a file. IoError if file exists.
    Create,
    /// Opens an existing file or creates a new one.
    OpenOrCreate,
    /// Opens an existing file or creates a new one, positioned at EOF.
    Append,
    /// Opens an existing file, truncating it to 0 bytes.
    Truncate,
    /// Opens an existing file or creates a new one, truncating it to 0 bytes.
    CreateOrTruncate,
}

enum FileAccess {
    Read,
    Write,
    ReadWrite
}

pub struct FileStream;

impl FileStream {
    pub fn open<P: PathLike>(_path: &P,
                             _mode: FileMode,
                             _access: FileAccess
                            ) -> Option<FileStream> {
        fail!()
    }
}

impl Reader for FileStream {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> {
        fail!()
    }

    fn eof(&mut self) -> bool {
        fail!()
    }
}

impl Writer for FileStream {
    fn write(&mut self, _v: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl Seek for FileStream {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

#[test]
#[ignore]
fn super_simple_smoke_test_lets_go_read_some_files_and_have_a_good_time() {
    let message = "it's alright. have a good time";
    let filename = &Path("test.txt");
    let mut outstream = FileStream::open(filename, Create, Read).unwrap();
    outstream.write(message.as_bytes());
}
