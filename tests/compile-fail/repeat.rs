fn main() {
    let data: [u8; std::usize::MAX] = [42; std::usize::MAX];
    //~^ ERROR: rustc layout computation failed: SizeOverflow([u8;
    assert_eq!(data.len(), 1024);
}
