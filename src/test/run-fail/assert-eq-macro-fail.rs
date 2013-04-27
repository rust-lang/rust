// error-pattern:left: 14 != right: 15

#[deriving(Eq)]
struct Point { x : int }

fn main() {
    assert_eq!(14,15);
}
