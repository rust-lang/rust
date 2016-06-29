//error-pattern: no mir for `std::env::args`

fn main() {
    let x = std::env::args();
    assert_eq!(x.count(), 1);
}
