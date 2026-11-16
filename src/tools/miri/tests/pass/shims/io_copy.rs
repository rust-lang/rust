//! io::copy on Linux has lots of specializations. Try to hit all relevant codepaths.
//@ignore-target: windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::fs;
use std::io::{self, Read, Seek, Write, pipe};

#[path = "../../utils/mod.rs"]
mod utils;

fn main() {
    test_copy_file_range();
    test_sendfile();
    test_splice();
}

fn test_copy_file_range() {
    // copy_file_range is used when src and dst are files.

    let bytes = b"123abc";
    let src_path = utils::prepare_with_content("miri_test_io_copy_src.txt", bytes);
    let dst_path = utils::prepare("miri_test_io_copy_dst.txt");

    let mut src = fs::File::open(&src_path).unwrap();
    let mut dst =
        fs::OpenOptions::new().create_new(true).read(true).write(true).open(&dst_path).unwrap();
    io::copy(&mut src, &mut dst).unwrap();

    // dst should be at end of file now
    let n = dst.read(&mut [0u8]).unwrap();
    assert_eq!(n, 0);
    // same for src
    assert_eq!(src.stream_position().unwrap(), bytes.len() as u64);

    // if we seek to the beginning we can find the contents
    dst.rewind().unwrap();
    let mut out = vec![];
    dst.read_to_end(&mut out).unwrap();
    assert_eq!(&*out, bytes);
}

fn test_sendfile() {
    // It is not possible to hit the sendfile path: the source has to be a file, and the destination
    // cannot be a fifo or socket. It also cannot be a file as then we use copy_file_range.
    // sendfile is therefore unreachable when copy_file_range is supported, and already
    // being tested above when copy_file_range is supported.
}

fn test_splice() {
    // splice is used when src or dst is a fifo, and (src is fifo or socket, and dest is file).
    // We go for a pipe-to-file copy.
    let bytes = b"123abc";
    let dst_path = utils::prepare("miri_test_io_copy_dst.txt");
    let mut dst =
        fs::OpenOptions::new().create_new(true).read(true).write(true).open(&dst_path).unwrap();

    let (mut pipe_rx, mut pipe_tx) = pipe().unwrap();
    pipe_tx.write_all(bytes).unwrap(); // fits into buffer, no need for a thread
    drop(pipe_tx);

    io::copy(&mut pipe_rx, &mut dst).unwrap();

    // dst should be at end of file now
    assert_eq!(dst.stream_position().unwrap(), bytes.len() as u64);

    // if we seek to the beginning we can find the contents
    dst.rewind().unwrap();
    let mut out = vec![];
    dst.read_to_end(&mut out).unwrap();
    assert_eq!(&*out, bytes);
}
