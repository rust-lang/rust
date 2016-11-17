fn main() {
    let data: [u8; 1024] = [42; 1024];
    assert_eq!(data.len(), 1024);
}
