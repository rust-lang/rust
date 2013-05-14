// error-pattern:left: 14 does not equal right: 15

#[deriving(Eq)]
struct Point { x : int }

fn main() {
    assert_eq!(14,15);
}
