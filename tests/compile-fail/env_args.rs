//error-pattern: no mir for `std

fn main() {
    let x = std::env::args();
    assert_eq!(x.count(), 1);
}
