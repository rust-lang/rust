//@only-target: apple # This directly tests apple-only functions

fn main() {
    let mut bytes = [0u8; 24];
    let ret = unsafe { libc::CCRandomGenerateBytes(bytes.as_mut_ptr().cast(), bytes.len()) };
    assert_eq!(ret, libc::kCCSuccess);
}
