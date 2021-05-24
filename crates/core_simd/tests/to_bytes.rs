use core_simd::SimdU32;

#[test]
fn byte_convert() {
    let int = SimdU32::from_array([0xdeadbeef, 0x8badf00d]);
    let bytes = int.to_ne_bytes();
    assert_eq!(int[0].to_ne_bytes(), bytes[..4]); 
    assert_eq!(int[1].to_ne_bytes(), bytes[4..]);
    assert_eq!(SimdU32::from_ne_bytes(bytes), int);
}
