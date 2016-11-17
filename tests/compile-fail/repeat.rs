fn main() {
    let data: [u8; std::isize::MAX as usize] = [42; std::isize::MAX as usize];
    //~^ ERROR: rustc layout computation failed: SizeOverflow([u8;
    assert_eq!(data.len(), 1024);
}
