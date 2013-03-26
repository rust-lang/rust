#[deriving(Eq)]
struct Point { x : int }

fn main() {
    assert_eq!(14,14);
    assert_eq!(~"abc",~"abc");
    assert_eq!(~Point{x:34},~Point{x:34});
    assert_eq!(&Point{x:34},&Point{x:34});
    assert_eq!(@Point{x:34},@Point{x:34});
}
