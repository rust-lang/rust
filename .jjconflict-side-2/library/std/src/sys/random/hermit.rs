pub fn fill_bytes(mut bytes: &mut [u8]) {
    while !bytes.is_empty() {
        let res = unsafe { hermit_abi::read_entropy(bytes.as_mut_ptr(), bytes.len(), 0) };
        assert_ne!(res, -1, "failed to generate random data");
        bytes = &mut bytes[res as usize..];
    }
}
