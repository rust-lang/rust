fn main() {
    let page_size = page_size::get();

    // In particular, this checks that it is not 0.
    assert!(page_size.is_power_of_two(), "page size not a power of two: {}", page_size);
}
