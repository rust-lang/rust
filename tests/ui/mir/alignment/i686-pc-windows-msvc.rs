//@ run-pass
//@ only-i686-pc-windows-msvc
//@ compile-flags: -Copt-level=0 -Cdebug-assertions=yes

// MSVC isn't sure if on 32-bit Windows its u64 type is 8-byte-aligned or 4-byte-aligned.
// So this test ensures that on i686-pc-windows-msvc, we do not insert a runtime check
// that will fail on dereferencing of a pointer to u64 which is not 8-byte-aligned but is
// 4-byte-aligned.

fn main() {
    let mut x = [0u64; 2];
    let ptr = x.as_mut_ptr();
    unsafe {
        let misaligned = ptr.byte_add(4);
        assert!(misaligned.addr() % 8 != 0);
        assert!(misaligned.addr() % 4 == 0);
        *misaligned = 42;
    }
}
