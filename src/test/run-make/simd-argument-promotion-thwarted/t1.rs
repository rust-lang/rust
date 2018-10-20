use std::arch::x86_64;

fn main() {
    if !is_x86_feature_detected!("avx2") {
        return println!("AVX2 is not supported on this machine/build.");
    }
    let load_bytes: [u8; 32] = [0x0f; 32];
    let lb_ptr = load_bytes.as_ptr();
    let reg_load = unsafe {
        x86_64::_mm256_loadu_si256(
            lb_ptr as *const x86_64::__m256i
        )
    };
    println!("{:?}", reg_load);
    let mut store_bytes: [u8; 32] = [0; 32];
    let sb_ptr = store_bytes.as_mut_ptr();
    unsafe {
        x86_64::_mm256_storeu_si256(sb_ptr as *mut x86_64::__m256i, reg_load);
    }
    assert_eq!(load_bytes, store_bytes);
}
