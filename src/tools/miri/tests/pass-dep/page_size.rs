fn main() {
    let page_size = page_size::get();

    // In particular, this checks that it is not 0.
    assert!(page_size.is_power_of_two(), "page size not a power of two: {}", page_size);
    // Most architectures have 4k pages by default
    #[cfg(not(any(
        target_arch = "wasm32",
        target_arch = "wasm64",
        all(target_arch = "aarch64", target_vendor = "apple")
    )))]
    assert!(page_size == 4 * 1024, "non-4k default page size: {}", page_size);
    // ... except aarch64-apple with 16k
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    assert!(page_size == 16 * 1024, "aarch64 apple reports non-16k page size: {}", page_size);
    // ... and wasm with 64k
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    assert!(page_size == 64 * 1024, "wasm reports non-64k page size: {}", page_size);
}
