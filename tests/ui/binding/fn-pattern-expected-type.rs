//@ run-pass

pub fn main() {
    let f = |(x, y): (isize, isize)| {
        assert_eq!(x, 1);
        assert_eq!(y, 2);
    };
    f((1, 2));
}
