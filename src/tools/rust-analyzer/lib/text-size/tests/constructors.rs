use text_size::TextSize;

#[derive(Copy, Clone)]
struct BadRope<'a>(&'a [&'a str]);

impl BadRope<'_> {
    fn text_len(self) -> TextSize {
        self.0.iter().copied().map(TextSize::of).sum()
    }
}

#[test]
fn main() {
    let x: char = 'c';
    let _ = TextSize::of(x);

    let x: &str = "hello";
    let _ = TextSize::of(x);

    let x: &String = &"hello".into();
    let _ = TextSize::of(x);

    let _ = BadRope(&[""]).text_len();
}
