//@ run-pass
// Test that an `&mut self` method, when invoked on a place whose
// type is `&mut [u8]`, passes in a pointer to the place and not a
// temporary. Issue #19147.

use std::slice;
use std::cmp;

trait MyWriter {
    fn my_write(&mut self, buf: &[u8]) -> Result<(), ()>;
}

impl<'a> MyWriter for &'a mut [u8] {
    fn my_write(&mut self, buf: &[u8]) -> Result<(), ()> {
        let amt = cmp::min(self.len(), buf.len());
        self[..amt].clone_from_slice(&buf[..amt]);

        let write_len = buf.len();
        unsafe {
            *self = slice::from_raw_parts_mut(
                self.as_mut_ptr().add(write_len),
                self.len() - write_len
            );
        }

        Ok(())
    }
}

fn main() {
    let mut buf = [0; 6];

    {
        let mut writer: &mut [_] = &mut buf;
        writer.my_write(&[0, 1, 2]).unwrap();
        writer.my_write(&[3, 4, 5]).unwrap();
    }

    // If `my_write` is not modifying `buf` in place, then we will
    // wind up with `[3, 4, 5, 0, 0, 0]` because the first call to
    // `my_write()` doesn't update the starting point for the write.

    assert_eq!(buf, [0, 1, 2, 3, 4, 5]);
}
