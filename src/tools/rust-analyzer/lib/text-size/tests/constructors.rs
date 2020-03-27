use {
    std::{borrow::Cow, sync::Arc},
    text_size::*,
};

#[derive(Copy, Clone)]
struct BadRope<'a>(&'a [&'a str]);

impl LenTextSize for BadRope<'_> {
    fn len_text_size(self) -> TextSize {
        self.0.iter().map(TextSize::of).sum()
    }
}

#[test]
fn main() {
    macro_rules! test {
        ($($expr:expr),+ $(,)?) => {
            $(let _ = TextSize::of($expr);)+
        };
    }

    test! {
        "",
        &"",
        'a',
        &'a',
        &String::new(),
        &String::new().into_boxed_str(),
        &Arc::new(String::new()),
        &Cow::Borrowed(""),
        BadRope(&[""]),
    }
}
