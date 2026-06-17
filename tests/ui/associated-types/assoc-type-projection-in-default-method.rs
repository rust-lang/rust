//@ run-pass

trait DigitCollection: Sized {
    type Iter: Iterator<Item = u8>;
    fn digit_iter(self) -> Self::Iter;

    fn digit_sum(self) -> u32 {
        self.digit_iter()
            .map(|digit: u8| digit as u32)
            .fold(0, |sum, digit| sum + digit)
    }
}

impl<I> DigitCollection for I where I: Iterator<Item=u8> {
    type Iter = I;

    fn digit_iter(self) -> I {
        self
    }
}

fn main() {
    let xs = vec![1, 2, 3, 4, 5];
    assert_eq!(xs.into_iter().digit_sum(), 15);
}
