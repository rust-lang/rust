// error-pattern the type `[u8;

fn main() {
    let data: [u8; std::usize::MAX] = [42; std::usize::MAX];
    assert_eq!(data.len(), 1024);
}
