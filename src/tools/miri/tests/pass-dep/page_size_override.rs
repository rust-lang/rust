//@compile-flags: -Zmiri-force-page-size=8

fn main() {
    let page_size = page_size::get();

    assert!(page_size == 8 * 1024, "8k page size override not respected: {}", page_size);
}
