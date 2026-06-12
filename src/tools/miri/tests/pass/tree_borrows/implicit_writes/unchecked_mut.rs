// This test reproduces the pattern used by `BorrowedCursor::as_mut`, which appears in `Socket::recv_with_flags` and `std::fs::read`.
// Many crates depend on similar patterns. Before https://github.com/rust-lang/rust/pull/157202 this failed under Tree
// Borrows with Implicit Writes. With the attribute `#[rustc_no_writable]` added to
// `slice::get_unchecked_mut`, both this test and the affected crates work.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

struct BorrowedBuf<'a> {
    buf: &'a mut [u8],
}

impl<'a> BorrowedBuf<'a> {
    fn capacity(&self) -> usize {
        self.buf.len()
    }

    unsafe fn as_mut(&mut self) -> &mut [u8] {
        unsafe { self.buf.get_unchecked_mut(..) }
    }
}

fn main() {
    let mut arr = [0u8; 4];
    let mut buf = BorrowedBuf { buf: &mut arr };

    let ptr = unsafe { buf.as_mut() }.as_mut_ptr();
    let _ = buf.capacity();
    unsafe { ptr.write(42); }
}
