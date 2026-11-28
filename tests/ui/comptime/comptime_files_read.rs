//@ compile-flags: -Z experimental-comptime-files=files/hello
//@ check-pass
#![feature(comptime_ops_fs)]

pub const fn assert_bytes_equal(a: &[u8], b: &[u8]) {
    assert!(a.len() == b.len());

    let mut idx = 0;
    while idx < a.len() {
        assert!(a[idx] == b[idx]);
        idx += 1;
    }
}

const _: () = {
    let file_content = core::mem::comptime::read_file("files/hello").unwrap();
    assert_bytes_equal(file_content, b"world");
};

fn main() {}
