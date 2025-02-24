//@ known-bug: #135570
//@compile-flags: -Zvalidate-mir -Zmir-enable-passes=+Inline -Copt-level=0 -Zmir-enable-passes=+GVN
//@ only-x86_64

fn function_with_bytes<const BYTES: &'static [u8; 0xc7b889180b67b07d_bc1a3c88783d35b5_u128]>(
) -> &'static [u8] {
    BYTES
}

fn main() {
    function_with_bytes::<b"aa">() == &[];
}
